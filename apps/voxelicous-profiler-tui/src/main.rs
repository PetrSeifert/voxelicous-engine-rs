//! TUI profiler dashboard for Voxelicous engine.
//!
//! Connects to a running Voxelicous application via TCP and displays
//! real-time profiling information.
//!
//! # Usage
//!
//! ```bash
//! cargo run -p voxelicous-profiler-tui
//! cargo run -p voxelicous-profiler-tui -- --host 127.0.0.1 --port 4242
//! ```

mod client;
mod ui;

use std::io;
use std::time::{Duration, Instant};

use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::prelude::*;

use client::{ConnectionState, ProfilerClient};
use ui::Dashboard;

/// Default host to connect to.
const DEFAULT_HOST: &str = "127.0.0.1";

/// Default port to connect to.
const DEFAULT_PORT: u16 = voxelicous_profiler::DEFAULT_PORT;

/// Target refresh rate for the TUI.
const REFRESH_RATE: Duration = Duration::from_millis(16); // ~60fps

/// Reconnection attempt interval.
const RECONNECT_INTERVAL: Duration = Duration::from_secs(2);

fn main() -> anyhow::Result<()> {
    // Parse command line arguments
    let (host, port) = parse_args();

    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Run the app
    let result = run_app(&mut terminal, &host, port);

    // Restore terminal
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    result
}

fn parse_args() -> (String, u16) {
    let args: Vec<String> = std::env::args().collect();
    let mut host = DEFAULT_HOST.to_string();
    let mut port = DEFAULT_PORT;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--host" | "-h" => {
                if i + 1 < args.len() {
                    host = args[i + 1].clone();
                    i += 1;
                }
            }
            "--port" | "-p" => {
                if i + 1 < args.len() {
                    if let Ok(p) = args[i + 1].parse() {
                        port = p;
                    }
                    i += 1;
                }
            }
            "--help" => {
                println!("Voxelicous Profiler TUI");
                println!();
                println!("Usage: voxelicous-profiler-tui [OPTIONS]");
                println!();
                println!("Options:");
                println!("  -h, --host <HOST>  Host to connect to (default: 127.0.0.1)");
                println!("  -p, --port <PORT>  Port to connect to (default: 4242)");
                println!("      --help         Show this help message");
                std::process::exit(0);
            }
            _ => {}
        }
        i += 1;
    }

    (host, port)
}

fn run_app(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    host: &str,
    port: u16,
) -> anyhow::Result<()> {
    let mut client = ProfilerClient::new(host, port);
    let dashboard = Dashboard::new(host, port);

    let mut last_reconnect = Instant::now() - RECONNECT_INTERVAL;

    loop {
        // Try to connect if disconnected
        if client.state() == ConnectionState::Disconnected {
            if last_reconnect.elapsed() >= RECONNECT_INTERVAL {
                let _ = client.connect();
                last_reconnect = Instant::now();
            }
        }

        // Poll for new data
        client.poll();

        // Draw the UI
        terminal.draw(|frame| {
            dashboard.render(frame, client.state(), client.last_snapshot());
        })?;

        // Handle input
        if event::poll(REFRESH_RATE)? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    match key.code {
                        KeyCode::Char('q') | KeyCode::Char('Q') | KeyCode::Esc => {
                            client.disconnect();
                            return Ok(());
                        }
                        KeyCode::Char('r') | KeyCode::Char('R') => {
                            let _ = client.send_reset();
                        }
                        KeyCode::Char('c') | KeyCode::Char('C') => {
                            client.disconnect();
                            let _ = client.connect();
                            last_reconnect = Instant::now();
                        }
                        _ => {}
                    }
                }
            }
        }
    }
}
