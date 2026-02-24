use std::path::PathBuf;

use anyhow::{bail, Result};
use clap::{Parser, Subcommand};
use genai::adapter::AdapterKind;
use genai::resolver::{AuthData, Endpoint, ServiceTargetResolver};
use genai::{Client, ModelIden, ServiceTarget};

use axon::mcp::McpServer;
use axon::rlm::{Rlm, RlmConfig};
use axon::store::ContextStore;

#[derive(Parser)]
#[command(name = "axon", about = "Recursive Language Model engine — one context, run everywhere")]
struct Cli {
    /// Root LLM model
    #[arg(long, default_value = "hf:minimax/minimax-m2.5")]
    model: String,

    /// Sub-RLM model (used for recursive calls)
    #[arg(long, default_value = "hf:minimax/minimax-m2.5")]
    sub_model: String,

    /// Custom provider base URL (e.g. for Synthetic/MiniMax)
    #[arg(long)]
    base_url: Option<String>,

    /// API key (overrides env var auto-detection)
    #[arg(long, env = "AXON_API_KEY", hide_env_values = true)]
    api_key: Option<String>,

    /// Max iterations per RLM level
    #[arg(long, default_value_t = 10)]
    max_iterations: usize,

    /// Max recursion depth for sub-RLM calls (0 = direct LLM, no sandbox)
    #[arg(long, default_value_t = 2)]
    max_depth: usize,

    /// Data directory for persistent context
    #[arg(long, default_value = "data")]
    data_dir: PathBuf,

    /// Verbose logging to stderr
    #[arg(short, long)]
    verbose: bool,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
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

    /// Run as an MCP stdio server
    Serve,
}

fn build_client(base_url: Option<&str>, api_key: Option<&str>) -> Client {
    let mut builder = Client::builder();

    if let Some(url) = base_url {
        let url = url.to_string();
        let key = api_key.map(|k| k.to_string());

        let resolver = ServiceTargetResolver::from_resolver_fn(
            move |target: ServiceTarget| {
                Ok(ServiceTarget {
                    endpoint: Endpoint::from_owned(url.clone()),
                    auth: key
                        .as_ref()
                        .map(|k| AuthData::from_single(k.clone()))
                        .unwrap_or(target.auth),
                    model: ModelIden::new(AdapterKind::OpenAI, target.model.model_name),
                })
            },
        );
        builder = builder.with_service_target_resolver(resolver);
    } else if let Some(key) = api_key {
        let key = key.to_string();
        let resolver = ServiceTargetResolver::from_resolver_fn(
            move |mut target: ServiceTarget| {
                target.auth = AuthData::from_single(key.clone());
                Ok(target)
            },
        );
        builder = builder.with_service_target_resolver(resolver);
    }

    builder.build()
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    let client = build_client(cli.base_url.as_deref(), cli.api_key.as_deref());
    let store = ContextStore::new(&cli.data_dir);

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

            let rlm = make_rlm(&client, &cli, 0);
            let answer = rlm.completion(query, &ctx).await?;
            println!("{}", answer);
        }

        Commands::Chat { ref thread } => {
            eprintln!("Axon RLM — thread '{}'. Type 'exit' to quit.", thread);
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
                let rlm = make_rlm(&client, &cli, 0);

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

        Commands::Store { ref file, ref thread } => {
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
            eprintln!(
                "Appended {} chars to thread '{}'.",
                text.len(),
                thread
            );
        }

        Commands::Serve => {
            let server = McpServer::new(
                client,
                cli.model.clone(),
                cli.sub_model.clone(),
                cli.max_iterations,
                cli.max_depth,
                store,
                cli.verbose,
            );
            server.run().await?;
        }
    }

    Ok(())
}

fn make_rlm(client: &Client, cli: &Cli, depth: usize) -> Rlm {
    Rlm::new(RlmConfig {
        client: client.clone(),
        model: cli.model.clone(),
        sub_model: cli.sub_model.clone(),
        max_iterations: cli.max_iterations,
        depth,
        max_depth: cli.max_depth,
        verbose: cli.verbose,
    })
}
