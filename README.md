# Axon

***One context, run everywhere.***

**Recursive Language Model engine in Rust — sandboxed Python execution via [ouros](https://github.com/parcadei/ouros), multi-provider LLM support via [genai](https://github.com/jeremychone/rust-genai).**

Built on the [Recursive Language Model](https://arxiv.org/abs/2512.24601v1) framework where LLMs offload context into a REPL environment and recursively call sub-LLMs to decompose complex tasks.

Axon uses **ouros** (a sandboxed Python runtime in Rust) for safe code execution and **genai** for unified access to OpenAI, Anthropic, Gemini, Ollama, and custom providers. When the RLM calls `llm_query()`, it spawns a **full sub-RLM at the next depth level** — each with its own sandbox — enabling true recursive reasoning.

## Architecture

```
Claude Code / User
  │
  └─ MCP (stdio) or CLI
      │
      ▼
Axon RLM Engine (Rust)
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

### Usage

**One-shot query against a context file:**
```bash
cargo run -- --base-url https://api.synthetic.com/v1 query "What is the magic number?" --context path/to/context.txt
```

**Interactive chat with persistent thread context:**
```bash
cargo run -- chat --thread myproject
```

**Store context for later queries:**
```bash
cargo run -- store transcript.txt --thread myproject
cat session.txt | cargo run -- store - --thread myproject
```

**Run as MCP server (for Claude Code):**
```bash
cargo run -- serve
```

### Claude Code Integration

```bash
claude mcp add axon --transport stdio -- /path/to/target/release/axon serve
```

## CLI Reference

```
axon [OPTIONS] <COMMAND>

Options:
  --model <MODEL>           Root LLM model [default: hf:minimax/minimax-m2.5]
  --sub-model <MODEL>       Sub-RLM model [default: hf:minimax/minimax-m2.5]
  --base-url <URL>          Custom provider base URL (for Synthetic/MiniMax/etc.)
  --api-key <KEY>           API key (or set AXON_API_KEY env var)
  --max-iterations <N>      Max iterations per RLM level [default: 10]
  --max-depth <N>           Max recursion depth [default: 2]
  --data-dir <DIR>          Data directory for context [default: data]
  -v, --verbose             Verbose logging to stderr

Commands:
  query   One-shot query against a context file
  chat    Interactive chat with persistent thread context
  store   Append text to a thread's context store
  serve   Run as an MCP stdio server
```

## Multi-Provider Support

Axon uses the [genai](https://github.com/jeremychone/rust-genai) crate for LLM access. Providers are auto-detected from model names:

| Model prefix | Provider | Env var |
|---|---|---|
| `gpt-*`, `o1*`, `o3*` | OpenAI | `OPENAI_API_KEY` |
| `claude*` | Anthropic | `ANTHROPIC_API_KEY` |
| `gemini*` | Google | `GEMINI_API_KEY` |
| `grok*` | xAI | `XAI_API_KEY` |
| Custom/unknown | Ollama (local) | — |

For custom providers (like Synthetic), use `--base-url` to set the endpoint:
```bash
export AXON_API_KEY=sk-...
axon --base-url https://api.synthetic.com/v1 --model minimax query "hello"
```

## MCP Tools

### `chat_rlm_query`

Query the RLM with persistent thread context.

| Param | Type | Description |
|-------|------|-------------|
| `query` | string | The question to ask |
| `thread_id` | string | Thread identifier — context accumulates per thread |

### `upload_context`

Upload a transcript to persistent memory.

| Param | Type | Description |
|-------|------|-------------|
| `transcript` | string | Full transcript text |
| `session_id` | string | Session identifier |
| `thread_id` | string | Thread to store under (default: `transcripts`) |

## Project Structure

```
axon/
├── Cargo.toml
├── src/
│   ├── main.rs        # CLI entry point (query, chat, store, serve)
│   ├── lib.rs         # Module declarations
│   ├── rlm.rs         # Core recursive RLM engine
│   ├── sandbox.rs     # ouros sandbox wrapper
│   ├── llm.rs         # genai-backed LLM client
│   ├── prompts.rs     # System prompts for root/sub RLMs
│   ├── store.rs       # Local filesystem context store
│   └── mcp.rs         # MCP stdio server
├── tests/
│   └── integration.rs # Integration tests
└── data/              # Persistent context (created at runtime)
```

## References

- [Recursive Language Models](https://arxiv.org/abs/2512.24601v1) — Zhang, Kraska & Khattab (2025)
- [ouros](https://github.com/parcadei/ouros) — Sandboxed Python runtime in Rust
- [genai](https://github.com/jeremychone/rust-genai) — Multi-AI Providers Library for Rust
- [Model Context Protocol](https://modelcontextprotocol.io)
