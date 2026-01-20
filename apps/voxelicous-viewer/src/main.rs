//! Voxelicous Engine Demo Viewer
//!
//! Renders procedurally generated terrain using ray tracing.
//! Automatically selects hardware RT (RTX) or compute ray marching based on GPU capabilities.
//!
//! ## Usage
//!
//! ```bash
//! cargo run -p voxelicous-viewer -- [OPTIONS]
//! ```
//!
//! ## Options
//!
//! - `-S, --screenshot`: Enable screenshot capture mode
//! - `-o, --output <PATTERN>`: Output path pattern (use `{}` for frame number)
//! - `-f, --frames <FRAMES>`: Frame indices to capture (e.g., "0,10,20" or "0-5")
//! - `--exit-after`: Exit after capturing all specified frames
//! - `-h, --help`: Print help message
//!
//! ## Examples
//!
//! ```bash
//! # Basic viewer
//! cargo run -p voxelicous-viewer
//!
//! # Capture frame 0
//! cargo run -p voxelicous-viewer -- -S
//!
//! # Capture frames 0, 10, 20 and exit
//! cargo run -p voxelicous-viewer -- -S -f 0,10,20 --exit-after
//!
//! # Capture frames 0-5 with custom output pattern
//! cargo run -p voxelicous-viewer -- -S -o frame_{}.png -f 0-5
//! ```
//!
//! ## Environment Variables
//!
//! - `VOXELICOUS_FORCE_COMPUTE`: Force compute ray marching (skip hardware RT)
//! - `RUST_LOG`: Set log level (e.g., info, debug, trace)

mod app;
mod terrain;

use voxelicous_app::{run_app, AppConfig};

use crate::app::Viewer;

const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;
const TARGET_FPS: u32 = 980;

fn main() -> anyhow::Result<()> {
    // Check for help flag before starting the app
    if std::env::args().any(|arg| arg == "-h" || arg == "--help") {
        print_help();
        return Ok(());
    }

    run_app::<Viewer>(
        AppConfig::new("Voxelicous Engine - Terrain Demo")
            .with_size(WIDTH, HEIGHT)
            .with_target_fps(TARGET_FPS),
    )
}

fn print_help() {
    eprintln!(
        "Voxelicous Engine Demo Viewer

USAGE:
    cargo run -p voxelicous-viewer -- [OPTIONS]

OPTIONS:
    -S, --screenshot        Enable screenshot capture mode
    -o, --output <PATTERN>  Output path pattern (use {{}} for frame number)
                            Default: screenshot_{{}}.png
    -f, --frames <FRAMES>   Frame indices to capture
                            Examples: \"0\" \"0,10,20\" \"0-5\" \"0,5-10,20\"
                            Default: 0
    --exit-after            Exit after capturing all specified frames
    -h, --help              Print this help message

EXAMPLES:
    # Capture frame 0
    cargo run -p voxelicous-viewer -- -S

    # Capture frames 0, 10, 20 and exit
    cargo run -p voxelicous-viewer -- -S -f 0,10,20 --exit-after

    # Capture frames 0-5 with custom output pattern
    cargo run -p voxelicous-viewer -- -S -o frame_{{}}.png -f 0-5

ENVIRONMENT VARIABLES:
    VOXELICOUS_FORCE_COMPUTE    Force compute ray marching (skip hardware RT)
    RUST_LOG                    Set log level (e.g., info, debug, trace)"
    );
}
