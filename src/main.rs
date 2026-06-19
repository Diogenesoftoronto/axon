use std::net::SocketAddr;
use std::path::PathBuf;

use anyhow::{bail, Result};
use clap::{Parser, Subcommand};
use genai::adapter::AdapterKind;
use genai::resolver::{AuthData, AuthResolver, Endpoint, ServiceTargetResolver};
use genai::{Client, ModelIden, ServiceTarget};

use altum::export::{export_sft, ExportOptions, ExportStats};
use altum::mcp::McpServer;
use altum::openai::OpenAiServer;
use altum::policy::{DepthEnforcementMode, PolicyCatalog, RuntimePolicy};
use altum::rlm::{Rlm, RlmConfig};
use altum::store::ContextStore;
use altum::tools::ToolRegistry;
use altum::trajectory::TrajectoryRecorder;

#[derive(Debug, Parser)]
#[command(
    name = "altum",
    about = "Recursive Language Model engine - one context, run everywhere"
)]
struct Cli {
    /// Root LLM model
    #[arg(long, default_value = "anthropic/claude-opus-4-6")]
    model: String,

    /// Sub-RLM model (used for recursive calls)
    #[arg(long, default_value = "anthropic/claude-opus-4-6")]
    sub_model: String,

    /// OpenAI-compatible provider base URL
    #[arg(
        long,
        env = "ALTUM_BASE_URL",
        default_value = "https://bifrost.dio.computer/v1/"
    )]
    base_url: Option<String>,

    /// API key (overrides env var auto-detection)
    #[arg(long, env = "ALTUM_API_KEY", hide_env_values = true)]
    api_key: Option<String>,

    /// Max iterations per RLM level
    #[arg(long, default_value_t = 10)]
    max_iterations: usize,

    /// Max recursion depth for sub-RLM calls (0 = direct LLM, no sandbox)
    #[arg(long, default_value_t = 1)]
    max_depth: usize,

    /// Data directory for persistent context
    #[arg(long, default_value = "data")]
    data_dir: PathBuf,

    /// Verbose logging to stderr
    #[arg(short, long)]
    verbose: bool,

    /// Trace sandbox execution steps (code blocks, external calls, vars) to stderr
    #[arg(long)]
    trace_sandbox: bool,

    /// Optional path to append per-run trajectory JSONL records
    #[arg(long)]
    trace_output: Option<PathBuf>,

    /// Prompt policy profile name
    #[arg(long, default_value = "baseline")]
    policy_profile: String,

    /// Prompt policy config path
    #[arg(long, default_value = "config/prompt_policies.json")]
    policy_config: PathBuf,

    /// Prepend active policy text into context as an option
    #[arg(long)]
    inject_policy_into_context: bool,

    /// Depth-enforcement mode
    #[arg(long, value_enum, default_value_t = DepthEnforcementMode::Off)]
    depth_enforcement: DepthEnforcementMode,

    /// Strict mode minimum depth threshold (absolute depth)
    #[arg(long)]
    require_min_depth: Option<usize>,

    /// Strict mode minimum recursive llm_query call count
    #[arg(long)]
    require_min_recursive_calls: Option<usize>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// One-shot query against a context file
    Query {
        /// The question to ask
        query: String,

        /// Path to context file
        #[arg(short, long)]
        context: Option<PathBuf>,

        /// Thread ID (reads from data dir)
        #[arg(short, long)]
        thread: Option<String>,
    },

    /// Interactive chat with persistent thread context
    Chat {
        /// Thread identifier (context accumulates per thread)
        #[arg(short, long, default_value = "default")]
        thread: String,
    },

    /// Append text to a thread's context store
    Store {
        /// File to read (or - for stdin)
        file: String,

        /// Thread to store under
        #[arg(short, long, default_value = "transcripts")]
        thread: String,
    },

    /// Run as an MCP server over stdio or streamable HTTP
    Serve {
        /// MCP transport to serve
        #[arg(long, value_enum, default_value_t = McpTransport::Stdio)]
        transport: McpTransport,

        /// Address to bind for HTTP transport
        #[arg(long, default_value = "127.0.0.1:3000")]
        bind: SocketAddr,

        /// Route path for HTTP transport
        #[arg(long, default_value = "/mcp")]
        path: String,
    },

    /// Run an OpenAI-compatible HTTP server for agent base_url integrations
    #[command(name = "serve-openai", alias = "serve-open-ai")]
    ServeOpenAi {
        /// Address to bind for the OpenAI-compatible HTTP server
        #[arg(long, default_value = "127.0.0.1:3000")]
        bind: SocketAddr,

        /// OpenAI-compatible models endpoint to expose from /v1/models
        #[arg(long, env = "ALTUM_MODELS_URL")]
        models_url: Option<String>,
    },

    /// Convert recorded trajectory JSONL into SFT finetuning data (OpenAI messages).
    #[command(name = "export-sft")]
    ExportSft {
        /// Path to input trajectory JSONL (one trajectory per line)
        #[arg(long)]
        input: PathBuf,

        /// Path to output SFT JSONL
        #[arg(long)]
        output: PathBuf,

        /// SFT format (only `openai` / `openai-messages` / `messages` supported)
        #[arg(long, default_value = "openai")]
        format: String,

        /// Drop trajectories whose final answer is empty
        #[arg(long, default_value_t = true)]
        require_final: bool,

        /// Minimum number of internal messages per kept trajectory
        #[arg(long, default_value_t = 2)]
        min_messages: usize,

        /// Cap the number of internal messages per kept trajectory
        #[arg(long)]
        max_messages: Option<usize>,

        /// Truncate each message content to at most N characters
        #[arg(long)]
        max_chars_per_message: Option<usize>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
enum McpTransport {
    Stdio,
    StreamableHttp,
}

fn build_client(base_url: Option<&str>, api_key: Option<&str>) -> Client {
    let mut builder = Client::builder();

    if let Some(url) = base_url {
        let url = url.to_string();
        let key = api_key.map(|k| k.to_string());

        let auth_resolver = key.as_ref().map(|k| {
            let k = k.clone();
            AuthResolver::from_resolver_fn(move |_model: ModelIden| {
                Ok(Some(AuthData::from_single(k.clone())))
            })
        });

        let target_resolver =
            ServiceTargetResolver::from_resolver_fn(move |target: ServiceTarget| {
                Ok(ServiceTarget {
                    endpoint: Endpoint::from_owned(url.clone()),
                    auth: key
                        .as_ref()
                        .map(|k| AuthData::from_single(k.clone()))
                        .unwrap_or(target.auth),
                    model: ModelIden::new(AdapterKind::OpenAI, target.model.model_name),
                })
            });

        if let Some(ar) = auth_resolver {
            builder = builder.with_auth_resolver(ar);
        }
        builder = builder.with_service_target_resolver(target_resolver);
    } else if let Some(key) = api_key {
        let key = key.to_string();
        let resolver = ServiceTargetResolver::from_resolver_fn(move |mut target: ServiceTarget| {
            target.auth = AuthData::from_single(key.clone());
            Ok(target)
        });
        builder = builder.with_service_target_resolver(resolver);
    }

    builder.build()
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    let client = build_client(cli.base_url.as_deref(), cli.api_key.as_deref());
    let store = ContextStore::new(&cli.data_dir);
    let policy_catalog = PolicyCatalog::load(&cli.policy_config);
    let default_runtime_policy = policy_catalog.build_runtime_policy(
        Some(&cli.policy_profile),
        Some(cli.inject_policy_into_context),
        Some(cli.depth_enforcement),
        cli.require_min_depth,
        cli.require_min_recursive_calls,
    );

    match cli.command {
        Commands::Query {
            ref query,
            ref context,
            ref thread,
        } => {
            let ctx = match (context, thread) {
                (Some(path), _) => std::fs::read_to_string(path)?,
                (None, Some(tid)) => store.read_context(tid),
                (None, None) => bail!("Provide --context <file> or --thread <id>"),
            };

            let rlm = make_rlm(&client, &cli, 0, default_runtime_policy.clone());
            let answer = rlm.completion(query, &ctx).await?;
            println!("{}", answer);
        }

        Commands::Chat { ref thread } => {
            eprintln!("Altum RLM - thread '{}'. Type 'exit' to quit.", thread);
            let stdin = std::io::stdin();
            loop {
                eprint!("You: ");
                let mut line = String::new();
                if stdin.read_line(&mut line)? == 0 {
                    break;
                }
                let query = line.trim();
                if query.is_empty() {
                    continue;
                }
                if matches!(query, "exit" | "quit" | ":q") {
                    eprintln!("Goodbye!");
                    break;
                }

                let ctx = store.read_context(thread);
                let rlm = make_rlm(&client, &cli, 0, default_runtime_policy.clone());

                eprintln!("Thinking...");
                match rlm.completion(query, &ctx).await {
                    Ok(answer) => {
                        println!("Assistant: {}", answer);
                        let _ = store.append_context(
                            thread,
                            &format!("\nUSER: {}\nASSISTANT: {}\n", query, answer),
                        );
                    }
                    Err(e) => eprintln!("Error: {}", e),
                }
            }
        }

        Commands::Store {
            ref file,
            ref thread,
        } => {
            let text = if file == "-" {
                let mut buf = String::new();
                std::io::Read::read_to_string(&mut std::io::stdin(), &mut buf)?;
                buf
            } else {
                std::fs::read_to_string(file)?
            };

            if text.trim().is_empty() {
                bail!("No content to store.");
            }

            store.append_context(thread, &text)?;
            eprintln!("Appended {} chars to thread '{}'.", text.len(), thread);
        }

        Commands::Serve {
            transport,
            ref path,
            bind,
        } => {
            let server = McpServer::new(
                client,
                cli.model.clone(),
                cli.sub_model.clone(),
                cli.max_iterations,
                cli.max_depth,
                store,
                cli.verbose,
                cli.trace_sandbox,
                policy_catalog,
                default_runtime_policy,
            );
            match transport {
                McpTransport::Stdio => server.serve_stdio().await?,
                McpTransport::StreamableHttp => server.serve_http(bind, path).await?,
            }
        }

        Commands::ServeOpenAi { bind, models_url } => {
            let models_url =
                models_url.or_else(|| cli.base_url.as_deref().map(models_url_for_base_url));
            let server = OpenAiServer::new(
                client,
                cli.model.clone(),
                cli.sub_model.clone(),
                cli.max_iterations,
                cli.max_depth,
                store,
                cli.verbose,
                cli.trace_sandbox,
                policy_catalog,
                default_runtime_policy,
                models_url,
            );
            eprintln!("Altum OpenAI-compatible server listening on http://{bind}/v1");
            server.serve(bind).await?;
        }

        Commands::ExportSft {
            ref input,
            ref output,
            ref format,
            require_final,
            min_messages,
            max_messages,
            max_chars_per_message,
        } => {
            let fmt = altum::export::SftFormat::parse(format)
                .ok_or_else(|| anyhow::anyhow!("unsupported --format '{}'", format))?;
            let opts = ExportOptions {
                format: fmt,
                require_final,
                min_messages,
                max_messages,
                max_chars_per_message,
            };
            let stats: ExportStats = export_sft(input, output, &opts)?;
            eprintln!(
                "ExportSft: read={} kept={} skipped_no_final={} skipped_too_few_messages={} skipped_truncated={} -> {}",
                stats.read,
                stats.kept,
                stats.skipped_no_final,
                stats.skipped_too_few_messages,
                stats.skipped_truncated_messages,
                output.display()
            );
        }
    }

    Ok(())
}

fn make_rlm(client: &Client, cli: &Cli, depth: usize, runtime_policy: RuntimePolicy) -> Rlm {
    let trace_sink = cli
        .trace_output
        .as_ref()
        .map(TrajectoryRecorder::to_path)
        .transpose()
        .ok()
        .flatten()
        .map(std::sync::Arc::new);
    Rlm::new(RlmConfig {
        client: client.clone(),
        model: cli.model.clone(),
        sub_model: cli.sub_model.clone(),
        max_iterations: cli.max_iterations,
        depth,
        max_depth: cli.max_depth,
        verbose: cli.verbose,
        trace_sandbox: cli.trace_sandbox,
        runtime_policy,
        tool_registry: std::sync::Arc::new(ToolRegistry::new()),
        trace_sink,
    })
}

fn models_url_for_base_url(base_url: &str) -> String {
    let trimmed = base_url.trim_end_matches('/');
    format!("{trimmed}/models")
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn test_serve_defaults_to_stdio() {
        let cli = Cli::parse_from(["altum", "serve"]);
        let Commands::Serve {
            transport,
            bind,
            path,
        } = cli.command
        else {
            panic!("expected serve command");
        };

        assert_eq!(transport, McpTransport::Stdio);
        assert_eq!(bind, "127.0.0.1:3000".parse::<SocketAddr>().unwrap());
        assert_eq!(path, "/mcp");
    }

    #[test]
    fn test_serve_accepts_streamable_http_flags() {
        let cli = Cli::parse_from([
            "altum",
            "--model",
            "hf:MiniMaxAI/MiniMax-M2.5",
            "serve",
            "--transport",
            "streamable-http",
            "--bind",
            "0.0.0.0:8080",
            "--path",
            "/custom",
        ]);

        let Commands::Serve {
            transport,
            bind,
            path,
        } = cli.command
        else {
            panic!("expected serve command");
        };

        assert_eq!(transport, McpTransport::StreamableHttp);
        assert_eq!(bind, "0.0.0.0:8080".parse::<SocketAddr>().unwrap());
        assert_eq!(path, "/custom");
    }

    #[test]
    fn test_invalid_transport_fails() {
        let err = Cli::try_parse_from(["altum", "serve", "--transport", "bogus"]).unwrap_err();
        assert!(err.to_string().contains("possible values"));
    }

    #[test]
    fn test_serve_openai_default_bind() {
        let cli = Cli::parse_from(["altum", "serve-openai"]);
        let Commands::ServeOpenAi { bind, models_url } = cli.command else {
            panic!("expected serve-openai command");
        };

        assert_eq!(bind, "127.0.0.1:3000".parse::<SocketAddr>().unwrap());
        assert_eq!(models_url, None);
    }

    #[test]
    fn test_serve_open_ai_alias() {
        let cli = Cli::parse_from(["altum", "serve-open-ai"]);
        let Commands::ServeOpenAi { bind, .. } = cli.command else {
            panic!("expected serve-open-ai alias");
        };

        assert_eq!(bind, "127.0.0.1:3000".parse::<SocketAddr>().unwrap());
    }

    #[test]
    fn test_models_url_for_base_url() {
        assert_eq!(
            models_url_for_base_url("https://bifrost.dio.computer/v1/"),
            "https://bifrost.dio.computer/v1/models"
        );
    }
}
