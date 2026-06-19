use arc_swap::ArcSwap;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tokio::sync::Mutex as AsyncMutex;

// Minimal representation of ToolRegistry for benchmarking
#[derive(Clone, Default)]
struct FakeToolRegistry {
    tools: HashMap<String, String>,
}

impl FakeToolRegistry {
    fn register(&mut self, name: String, desc: String) {
        self.tools.insert(name, desc);
    }
}

async fn bench_mutex(users: usize, iterations: usize, updates: usize) {
    let registry = Arc::new(Mutex::new(FakeToolRegistry::default()));
    let start = Instant::now();

    let mut handles = vec![];

    // Reader tasks
    for _ in 0..users {
        let registry = registry.clone();
        handles.push(tokio::spawn(async move {
            for _ in 0..iterations {
                let _reg = registry.lock().unwrap().clone();
            }
        }));
    }

    // Writer task
    let registry_writer = registry.clone();
    handles.push(tokio::spawn(async move {
        for i in 0..updates {
            {
                let mut reg = registry_writer.lock().unwrap();
                reg.register(format!("tool-{}", i), "desc".to_string());
            }
            tokio::task::yield_now().await;
        }
    }));

    for handle in handles {
        handle.await.unwrap();
    }

    println!("Mutex:    {:?}", start.elapsed());
}

async fn bench_arc_swap(users: usize, iterations: usize, updates: usize) {
    let registry = Arc::new(ArcSwap::from_pointee(FakeToolRegistry::default()));
    let write_lock = Arc::new(AsyncMutex::new(()));
    let start = Instant::now();

    let mut handles = vec![];

    // Reader tasks
    for _ in 0..users {
        let registry = registry.clone();
        handles.push(tokio::spawn(async move {
            for _ in 0..iterations {
                let _reg = registry.load_full();
            }
        }));
    }

    // Writer task
    let registry_writer = registry.clone();
    handles.push(tokio::spawn(async move {
        for i in 0..updates {
            {
                let _guard = write_lock.lock().await;
                let mut new_reg = (**registry_writer.load()).clone();
                new_reg.register(format!("tool-{}", i), "desc".to_string());
                registry_writer.store(Arc::new(new_reg));
            }
            tokio::task::yield_now().await;
        }
    }));

    for handle in handles {
        handle.await.unwrap();
    }

    println!("ArcSwap:  {:?}", start.elapsed());
}

#[tokio::main]
async fn main() {
    let users = 100;
    let iterations = 1000;
    let updates = 10;

    println!(
        "Simulating {} users, {} reads each, {} updates total...",
        users, iterations, updates
    );

    // Warm up
    bench_mutex(1, 10, 0).await;
    bench_arc_swap(1, 10, 0).await;

    bench_mutex(users, iterations, updates).await;
    bench_arc_swap(users, iterations, updates).await;
}
