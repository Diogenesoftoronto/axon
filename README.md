# Altum

***One context, run everywhere.***

**Recursive Language Model engine in Rust - sandboxed Python execution via [ouros](https://github.com/parcadei/ouros), multi-provider LLM support via [genai](https://github.com/jeremychone/rust-genai).**

Built on the [Recursive Language Model](https://arxiv.org/abs/2512.24601v1) framework where LLMs offload context into a REPL environment and recursively call sub-LLMs to decompose complex tasks.

Altum uses **ouros** (a sandboxed Python runtime in Rust) for safe code execution and **genai** for unified access to OpenAI, Anthropic, Gemini, Ollama, and custom providers. When the RLM calls `llm_query()`, it spawns a **full sub-RLM at the next depth level** - each with its own sandbox - enabling true recursive reasoning.

## Architecture

```
Claude Code / User
  │
  └─ MCP (stdio or streamable HTTP) or CLI
      │
      ▼
Altum RLM Engine (Rust)
  │
  ├─ chat_rlm_query(query, thread_id)
  │     ├─ reads context from local filesystem: data/{thread_id}/context.txt
  │     ├─ runs RLM reasoning loop:
  │     │    root LLM ──writes code──▶ ouros sandbox (Python)
  │     │                               │
  │     │    sandbox calls llm_query() ─────▶ sub-RLM (depth+1, own sandbox)
  │     │                                      │
  │     │    sub-RLM calls llm_query() ─────▶ sub-sub-RLM or direct LLM
  │     │                                      │
  │     │    results flow back up ◀────────────┘
  │     │    ... repeat up to N iterations
  │     ├─ appends Q&A turn to local context
  │     └─ returns answer
  │
  └─ upload_context(transcript, session_id, thread_id)
        └─ appends transcript to: data/{thread_id}/context.txt
```

## Quick Start

### Prerequisites

- Rust toolchain (1.75+)
- An API key for your LLM provider

### Build

```bash
cargo build --release
```

### Install From crates.io

Once Altum is published to crates.io:

```bash
cargo install altum
```

Then use the installed `altum` binary directly:

```bash
altum --help
```

### Install From GitHub Releases

Download a prebuilt binary from the latest release:

```bash
gh release download --repo Diogenesoftoronto/altum --pattern "altum-*" --dir .
```

You can also download a specific version:

```bash
gh release download v0.1.0 --repo Diogenesoftoronto/altum --pattern "altum-*"
```

### Usage

**One-shot query against a context file:**
```bash
cargo run -- query "What is the magic number?" --context path/to/context.txt
# After publishing/installing:
altum query "What is the magic number?" --context path/to/context.txt
```

**Interactive chat with persistent thread context:**
```bash
cargo run -- chat --thread myproject
# After publishing/installing:
altum chat --thread myproject
```

**Store context for later queries:**
```bash
cargo run -- store transcript.txt --thread myproject
cat session.txt | cargo run -- store - --thread myproject
# After publishing/installing:
altum store transcript.txt --thread myproject
cat session.txt | altum store - --thread myproject
```

**Run as MCP server over stdio (for Claude Code):**
```bash
cargo run -- serve
# After publishing/installing:
altum serve
```

**Run as MCP server over streamable HTTP:**
```bash
cargo run -- serve --transport streamable-http --bind 127.0.0.1:3000 --path /mcp
# After publishing/installing:
altum serve --transport streamable-http --bind 127.0.0.1:3000 --path /mcp
```

**Run as an OpenAI-compatible server for agent `base_url` integrations:**
```bash
cargo run -- serve-openai --bind 127.0.0.1:3000
# After publishing/installing:
altum serve-openai --bind 127.0.0.1:3000
```

Then point an OpenAI-compatible agent at:

```text
http://127.0.0.1:3000/v1
```

Altum supports `POST /v1/chat/completions`, `POST /v1/responses`, and
`GET /v1/models`. Persistent RLM memory uses `metadata.thread_id`,
`X-Altum-Thread`, or the OpenAI `user` field, falling back to `default`.

By default, Altum connects to the Railway-hosted Bifrost gateway at
`https://bifrost.dio.computer/v1/` and mirrors
`https://bifrost.dio.computer/v1/models` from its own `/v1/models` endpoint.
Override this with `--base-url` and `--models-url` if you want to point Altum at
another OpenAI-compatible provider.

### Claude Code Integration

```bash
claude mcp add altum --transport stdio -- /path/to/target/release/altum serve
```

## CLI Reference

```
altum [OPTIONS] <COMMAND>

Options:
  --model <MODEL>           Root LLM model [default: anthropic/claude-sonnet-4-6]
  --sub-model <MODEL>       Sub-RLM model [default: anthropic/claude-haiku-4-5-20251001]
  --base-url <URL>          OpenAI-compatible provider base URL [default: https://bifrost.dio.computer/v1/]
  --api-key <KEY>           API key (or set ALTUM_API_KEY env var)
  --max-iterations <N>      Max iterations per RLM level [default: 10]
  --max-depth <N>           Max recursion depth [default: 2]
  --policy-profile <NAME>   Prompt policy profile [default: baseline]
  --policy-config <PATH>    Policy config JSON [default: config/prompt_policies.json]
  --inject-policy-into-context
                            Prepend active policy into runtime context
  --depth-enforcement <MODE>
                            Depth gate mode: off|soft|strict [default: off]
  --require-min-depth <N>   Strict mode: minimum depth reached
  --require-min-recursive-calls <N>
                            Strict mode: minimum recursive llm_query calls
  --data-dir <DIR>          Data directory for context [default: data]
  -v, --verbose             Verbose logging to stderr

Commands:
  query   One-shot query against a context file
  chat    Interactive chat with persistent thread context
  store   Append text to a thread's context store
  serve   Run as an MCP server over stdio or streamable HTTP
  serve-openai
          Run an OpenAI-compatible HTTP server for agent base_url integrations
```

`serve` options:

```text
  --transport <stdio|streamable-http>  MCP transport [default: stdio]
  --bind <HOST:PORT>                   Bind address for HTTP transport [default: 127.0.0.1:3000]
  --path <PATH>                        Route path for HTTP transport [default: /mcp]
```

`serve-openai` options:

```text
  --bind <HOST:PORT>                   Bind address [default: 127.0.0.1:3000]
  --models-url <URL>                   Models endpoint to expose from /v1/models
```

## Multi-Provider Support

Altum uses the [genai](https://github.com/jeremychone/rust-genai) crate for LLM access. Providers are auto-detected from model names:

| Model prefix | Provider | Env var |
|---|---|---|
| `gpt-*`, `o1*`, `o3*` | OpenAI | `OPENAI_API_KEY` |
| `claude*` | Anthropic | `ANTHROPIC_API_KEY` |
| `gemini*` | Google | `GEMINI_API_KEY` |
| `grok*` | xAI | `XAI_API_KEY` |
| Custom/unknown | Ollama (local) | - |

The default endpoint is the Railway-hosted Bifrost gateway:

```bash
export ALTUM_API_KEY=sk-...
altum query "hello"
```

For another OpenAI-compatible provider, use `--base-url` to set the endpoint:

```bash
export ALTUM_API_KEY=sk-...
altum --base-url https://api.synthetic.com/v1 --model minimax query "hello"
```

When serving Altum as an OpenAI-compatible endpoint, the models list is fetched
from `<base-url>/models` by default:

```bash
altum serve-openai \
  --bind 127.0.0.1:3000 \
  --models-url https://bifrost.dio.computer/v1/models
```

## MCP Tools

### `chat_rlm_query`

Query the RLM with persistent thread context.

| Param | Type | Description |
|-------|------|-------------|
| `query` | string | The question to ask |
| `thread_id` | string | Thread identifier - context accumulates per thread |
| `policy_profile` | string (optional) | Per-call policy profile override |
| `inject_policy_into_context` | boolean (optional) | Prepend policy text to context for this call |
| `depth_enforcement` | string (optional) | `off`, `soft`, or `strict` |
| `require_min_depth` | integer (optional) | Strict minimum depth threshold |
| `require_min_recursive_calls` | integer (optional) | Strict minimum recursive call threshold |

### `upload_context`

Upload a transcript to persistent memory.

| Param | Type | Description |
|-------|------|-------------|
| `transcript` | string | Full transcript text |
| `session_id` | string | Session identifier |
| `thread_id` | string (optional) | Thread to store under (default: `transcripts`) |

## Project Structure

```text
altum/
├── Cargo.toml
├── src/
│   ├── main.rs        # CLI entry point (query, chat, store, serve)
│   ├── lib.rs         # Module declarations
│   ├── rlm.rs         # Core recursive RLM engine
│   ├── sandbox.rs     # ouros sandbox wrapper
│   ├── llm.rs         # genai-backed LLM client
│   ├── prompts.rs     # System prompts for root/sub RLMs
│   ├── store.rs       # Local filesystem context store
│   └── mcp.rs         # rmcp-backed MCP server and transport wiring
├── tests/
│   └── integration.rs # Integration tests
└── data/              # Persistent context (created at runtime)
```

## Extending Altum

This project is small and intentionally modular. If you want to add features, use this path:

1. Add a new behavior in the engine (`src/rlm.rs`) and sandbox bridge (`src/sandbox.rs`).
2. Add a new CLI surface in [`main.rs`](src/main.rs) if users need direct access.
3. Add or update MCP tools in [`mcp.rs`](src/mcp.rs) if Claude Code should call it.
4. Update prompts in [`prompts.rs`](src/prompts.rs) when behavior depends on agent instructions.
5. Add unit tests next to the changed module and integration tests in [`tests/integration.rs`](tests/integration.rs).

Common extension patterns:

- Add a sandbox external function. Register it in `Sandbox::new()` in `src/sandbox.rs`.
- Add a sandbox external function. Handle it in `Rlm::handle_external()` in `src/rlm.rs`.
- Add a sandbox external function. Decide whether it should be callable from prompts and document it.
- Add a new CLI command. Extend the `Commands` enum in `src/main.rs`.
- Add a new CLI command. Implement the command branch in `main()`.
- Add a new MCP tool. Add a typed params struct and an `#[tool]` method in `src/mcp.rs`.
- Add a new MCP tool. Return `CallToolResult::success(...)` or `CallToolResult::error(...)` from the tool implementation.

Potential feature design docs:

- [Fork, Checkpoint, and VFS Extension](docs/fork-checkpoint-vfs.md)

Local validation before pushing:

```bash
cargo fmt --all --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-targets --all-features
cargo build --release
```

## Release Process

This repo publishes binaries from `.github/workflows/release.yml`.

1. Create a version tag.
2. Push the tag.
3. GitHub Actions builds binaries for Linux, macOS (Intel/Apple Silicon), and Windows.
4. Assets are attached to the GitHub Release automatically.

With `gh`:

```bash
git tag v0.1.0
git push origin v0.1.0
gh run list --repo Diogenesoftoronto/altum --workflow release.yml
gh release view v0.1.0 --repo Diogenesoftoronto/altum
```

## Benchmarking

The benchmark suite (`benchmarks/mode_profile_targeted.json`) contains 12 tasks across 5 categories, designed to align with the RLM and agent-scaling literature:

| Category | Tasks | Context | Aligned With |
|---|---|---|---|
| Cognitive traps | 2 | ~100 chars | Zhang 2025, Chen 2026 |
| State tracking | 1 | ~200 chars | Chen 2026 (sub-structure) |
| Word/delegation puzzles | 2 | ~100–200 chars | Kim 2025 (decomposability) |
| Multi-step reasoning | 3 | ~200–400 chars | Prime Intellect (math-python), Zhang (recursive) |
| Long-context distributional | 4 | 10K–15K chars | OOLONG, Prime Intellect (DeepDive) |

### Running benchmarks

```bash
export SYNTHETIC_API_KEY=...

# Quick single-model run
python3 scripts/benchmark_altum.py \
  --dataset benchmarks/mode_profile_targeted.json \
  --model "hf:MiniMaxAI/MiniMax-M2.5" \
  --pricing-from-models-api \
  --timeout 120

# Multi-model Pareto analysis (7 models × 5 modes × 12 tasks)
python3 scripts/benchmark_altum.py \
  --dataset benchmarks/mode_profile_targeted.json \
  --model-list benchmarks/models_pareto_live.txt \
  --pricing-from-models-api \
  --runs 1 \
  --attempts-per-run 3 \
  --timeout 180 \
  --out benchmarks/results/results-latest.json \
  --summary-md benchmarks/results/summary-latest.md

# Policy optimization from benchmark artifacts
python3 scripts/optimize_prompt_policy.py benchmarks/results/results-*.json
```

### Statistical analysis

```bash
# Bayesian posteriors, Thompson sampling, latency variability
python3 scripts/analyze_bayesian.py benchmarks/results/results-latest.json

# From in-progress benchmark log
python3 scripts/analyze_bayesian.py --from-log /tmp/benchmark.log

# Generate LaTeX snippets for paper
python3 scripts/analyze_bayesian.py benchmarks/results/results-latest.json --latex-out paper/stats.tex
```

### Documentation

- `docs/benchmarking.md` — benchmark harness reference
- `docs/testing-playbook.md` — standard testing scripts/commands
- `paper/main.tex` — research paper (compiles with tectonic/pdflatex)

## References

- [Recursive Language Models](https://arxiv.org/abs/2512.24601v1) - Zhang, Kraska & Khattab (2025)
- [ouros](https://github.com/parcadei/ouros) - Sandboxed Python runtime in Rust
- [genai](https://github.com/jeremychone/rust-genai) - Multi-AI Providers Library for Rust
- [Model Context Protocol](https://modelcontextprotocol.io)
