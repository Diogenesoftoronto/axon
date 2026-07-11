//! Per-runtime code repair helpers.
//!
//! When a runtime's parser rejects code the model emitted, the RLM has two
//! choices: reject the iteration outright or attempt a targeted repair and
//! retry. This module implements the targeted-repair variants.
//!
//! ## Workflow
//!
//! 1. The runtime attempts to execute the code block.
//! 2. On a syntax/parse error, [`RepairEngine::try_repair`] is invoked with
//!    the raw code and the error message.
//! 3. If a repair applies, the RLM retries execution with the rewritten code.
//! 4. Every applied repair is recorded for telemetry in
//!    [`RepairEngine::take_log`].
//!
//! Steel-specific repairs (designed to be conservative and only handle
//! commonly emitted issues):
//!
//! - unmatched parentheses (adds missing `)` / `]` / `}` until the source
//!   parses),
//! - incomplete final form (appends `# incomplete form` so the parser does
//!   not require more input),
//! - malformed string delimiters (re-closes unbalanced `"`),
//! - common Python idioms accidentally emitted in Scheme mode (e.g. `=`
//!   replaced with `define`).
//!
//! The Ouros runtime relies on ouros's own parser, which produces Ouroboros
//! diagnostics. The generic repair here is reserved for the Steel experiment
//! because Steel's parser surface is small and well-understood.

use std::sync::Mutex;

use anyhow::Result;

#[derive(Debug, Default)]
pub struct RepairEngine {
    log: Mutex<Vec<RepairEntry>>,
}

#[derive(Debug, Clone)]
pub struct RepairEntry {
    pub rule: RepairRule,
    pub snippet: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RepairRule {
    /// Balances unbalanced parentheses (`(`/`)`).
    BalanceParens,
    /// Closes an unterminated string literal.
    CloseString,
    /// Appends a niche closure to a trailing incomplete form.
    CloseIncompleteForm,
    /// Replaces a Python-style top-level `=` with a `define`.
    PythonEqualsToDefine,
}

impl RepairRule {
    pub fn label(self) -> &'static str {
        match self {
            RepairRule::BalanceParens => "balance-parens",
            RepairRule::CloseString => "close-string",
            RepairRule::CloseIncompleteForm => "close-incomplete-form",
            RepairRule::PythonEqualsToDefine => "python-equals->define",
        }
    }
}

impl RepairEngine {
    pub fn new() -> Self {
        Self::default()
    }

    /// Records an applied repair. Idempotent from the engine's perspective.
    pub fn record(&self, rule: RepairRule, snippet: String) {
        if let Ok(mut log) = self.log.lock() {
            log.push(RepairEntry { rule, snippet });
        }
    }

    /// Returns all recorded repairs since the last call and clears the log.
    pub fn take_log(&self) -> Vec<RepairEntry> {
        self.log
            .lock()
            .map(|mut log| std::mem::take(&mut *log))
            .unwrap_or_default()
    }

    /// Attempts to repair `code` for a runtime of the given language.
    ///
    /// Returns `Ok(Some(repaired))` if a repair rule matched, or
    /// `Ok(None)` if no rule applies. Diagnostic message is used as a hint
    /// but the current implementation is pattern-based on the source code.
    pub fn try_repair(
        &self,
        code: &str,
        error_message: &str,
        lang: &str,
    ) -> Result<Option<String>> {
        match lang {
            "scheme" => self.try_repair_scheme(code, error_message),
            _ => Ok(None),
        }
    }

    fn try_repair_scheme(&self, code: &str, error_message: &str) -> Result<Option<String>> {
        let mut result = String::from(code);
        let mut applied_any = false;

        // Rule 1: balance parentheses.
        if let Some(repaired) = balance_parens(&result) {
            self.record(RepairRule::BalanceParens, repaired.clone());
            result = repaired;
            applied_any = true;
        }

        // Rule 2: close unterminated strings.
        if let Some(repaired) = close_unterminated_strings(&result) {
            self.record(RepairRule::CloseString, repaired.clone());
            result = repaired;
            applied_any = true;
        }

        // Rule 3: Python-style `def foo(x):` or `x = 5` at the top level.
        if let Some(repaired) = python_top_level_define(&result) {
            self.record(RepairRule::PythonEqualsToDefine, repaired.clone());
            result = repaired;
            applied_any = true;
        }

        if applied_any {
            Ok(Some(result))
        } else {
            // Failed hints are useful for future rule design.
            let _ = error_message;
            Ok(None)
        }
    }
}

/// Balances unmatched `(`, `[`, and `{` by appending missing closers.
pub fn balance_parens(code: &str) -> Option<String> {
    let mut stack: Vec<char> = Vec::new();
    let mut in_string = false;
    let mut escaped = false;
    let mut in_comment = false;

    for ch in code.chars() {
        if in_comment {
            if ch == '\n' {
                in_comment = false;
            }
            continue;
        }
        if in_string {
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                in_string = false;
            }
            continue;
        }
        match ch {
            '"' => {
                in_string = true;
            }
            ';' => in_comment = true,
            '(' | '[' | '{' => stack.push(ch),
            ')' | ']' | '}' => {
                let closer_match = match ch {
                    ')' => '(',
                    ']' => '[',
                    '}' => '{',
                    _ => unreachable!(),
                };
                if stack.last() == Some(&closer_match) {
                    stack.pop();
                }
            }
            _ => {}
        }
    }

    if stack.is_empty() {
        return None;
    }

    let mut result = String::from(code);
    while let Some(open) = stack.pop() {
        result.push(match open {
            '(' => ')',
            '[' => ']',
            '{' => '}',
            other => other,
        });
    }
    Some(result)
}

/// Closes an unterminated Scheme string literal (`"`). Apostrophes are the
/// quote operator in Scheme, not string delimiters.
pub fn close_unterminated_strings(code: &str) -> Option<String> {
    let mut in_string = false;
    let mut escaped = false;
    let mut in_comment = false;
    for ch in code.chars() {
        if in_comment {
            if ch == '\n' {
                in_comment = false;
            }
            continue;
        }
        if in_string {
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                in_string = false;
            }
            continue;
        }
        match ch {
            '"' => in_string = true,
            ';' => in_comment = true,
            _ => {}
        }
    }
    if in_string {
        let mut result = String::from(code);
        result.push('"');
        Some(result)
    } else {
        None
    }
}

/// Replaces Python-style top-level `x = expr` with `(define x expr)`.
/// Conservative: only applies when the line starts with an identifier
/// followed by `=` and no surrounding parentheses.
pub fn python_top_level_define(code: &str) -> Option<String> {
    let lines: Vec<&str> = code.lines().collect();
    let mut transformed = Vec::new();
    let mut applied = false;
    for line in lines.iter() {
        let trimmed = line.trim_start();
        if trimmed.is_empty()
            || trimmed.starts_with(';')
            || trimmed.starts_with('(')
            || !trimmed.contains('=')
        {
            transformed.push((*line).to_string());
            continue;
        }
        // Quick shape-match: `<identifier> = <rest>`.
        let eq_idx = match trimmed.find('=') {
            Some(i) => i,
            None => {
                transformed.push((*line).to_string());
                continue;
            }
        };
        let lhs = trimmed[..eq_idx].trim();
        let rhs = trimmed[eq_idx + 1..].trim();
        if lhs
            .chars()
            .next()
            .map(|c| c.is_alphabetic() || c == '_')
            .unwrap_or(false)
            && !lhs.contains(' ')
            && !rhs.is_empty()
        {
            let indent_len = line.len() - trimmed.len();
            transformed.push(format!(
                "{}(define {} {})",
                " ".repeat(indent_len),
                lhs,
                rhs
            ));
            applied = true;
        } else {
            transformed.push((*line).to_string());
        }
    }
    if applied {
        Some(transformed.join("\n"))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn balance_parens_appends_missing_close() {
        let repaired = balance_parens("(+ 1 2").unwrap();
        assert_eq!(repaired, "(+ 1 2)");
    }

    #[test]
    fn balance_parens_handles_nested() {
        let repaired = balance_parens("(let ([x 1] [y 2]) (+ x y").unwrap();
        assert_eq!(repaired, "(let ([x 1] [y 2]) (+ x y))");
    }

    #[test]
    fn balance_parens_respects_strings() {
        let input = "(display \") mismatch\")";
        assert!(balance_parens(input).is_none());
    }

    #[test]
    fn balance_parens_tracks_escaped_quotes_at_the_current_position() {
        let input = r#"(display "a \"(""#;
        assert_eq!(balance_parens(input).unwrap(), r#"(display "a \"(")"#);
    }

    #[test]
    fn balance_parens_treats_apostrophe_as_quote_syntax() {
        let input = "(define x 'foo";
        assert_eq!(balance_parens(input).unwrap(), "(define x 'foo)");
    }

    #[test]
    fn balance_parens_respects_comments() {
        let input = "(define x 1) ; not closing this (";
        assert!(balance_parens(input).is_none());
    }

    #[test]
    fn close_unterminated_strings_appends_quote() {
        let repaired = close_unterminated_strings("(display \"hello)").unwrap();
        assert!(repaired.ends_with('"'));
    }

    #[test]
    fn close_unterminated_strings_ignores_scheme_quote_syntax() {
        assert!(close_unterminated_strings("(define x 'foo)").is_none());
    }

    #[test]
    fn close_unterminated_strings_respects_escaped_quotes() {
        let input = r#"(display "a \"quoted\""#;
        assert_eq!(
            close_unterminated_strings(input).unwrap(),
            format!("{input}\"")
        );
    }

    #[test]
    fn python_top_level_define_rewrites_basic() {
        let before = "x = 5";
        let after = python_top_level_define(before).unwrap();
        assert_eq!(after, "(define x 5)");
    }

    #[test]
    fn python_top_level_define_skips_already_parens() {
        let before = "(define y 6)";
        assert!(python_top_level_define(before).is_none());
    }

    #[test]
    fn repair_engine_records_applied_rules() {
        let engine = RepairEngine::new();
        let result = engine
            .try_repair("(+ 1 2", "unexpected EOF", "scheme")
            .unwrap();
        assert!(result.is_some());
        assert_eq!(engine.take_log().len(), 1);
    }

    #[test]
    fn repair_engine_no_match_for_unsupported_lang() {
        let engine = RepairEngine::new();
        let result = engine.try_repair("(+ 1 2", "broken", "repl").unwrap();
        assert!(result.is_none());
    }
}
