//! Voxelicous Engine Demo Viewer
//!
//! Renders procedurally generated terrain using clipmap compute ray marching.
//!
//! ## Usage
//!
//! ```bash
//! cargo run -p voxelicous-viewer -- [OPTIONS]
//! ```
//!
//! ## Options
//!
//! ### Screenshot options
//! - `-S, --screenshot`: Enable screenshot capture mode
//! - `-o, --output <PATTERN>`: Output path pattern (use `{}` for frame number)
//! - `-f, --frames <FRAMES>`: Frame indices to capture (e.g., "0,10,20" or "0-5")
//! - `--exit-after`: Exit after capturing all specified frames
//!
//! ### World options
//! - `--seed <N>`: World generation seed (default: 42)
//!
//! ### Other
//! - `-h, --help`: Print help message
//!
//! ## Examples
//!
//! ```bash
//! # Basic viewer
//! cargo run -p voxelicous-viewer
//!
//! # Use a custom seed
//! cargo run -p voxelicous-viewer -- --seed 1234
//!
//! # Capture frame 0
//! cargo run -p voxelicous-viewer -- -S
//!
//! # Capture frames during orbit and exit
//! cargo run -p voxelicous-viewer -- -S -f 0,50,100,150,200 -o stream_{}.png --exit-after
//! ```
//!
//! ## Environment Variables
//!
//! - `RUST_LOG`: Set log level (e.g., info, debug, trace)

mod app;

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
        AppConfig::new("Voxelicous Engine - Clipmap Demo")
            .with_size(WIDTH, HEIGHT)
            .with_target_fps(TARGET_FPS),
    )
}

fn print_help() {
    eprintln!(
        "Voxelicous Engine Demo Viewer with Clipmap Ray Marching

USAGE:
    cargo run -p voxelicous-viewer -- [OPTIONS]

SCREENSHOT OPTIONS:
    -S, --screenshot        Enable screenshot capture mode
    -o, --output <PATTERN>  Output path pattern (use {{}} for frame number)
                            Default: screenshot_{{}}.png
    -f, --frames <FRAMES>   Frame indices to capture
                            Examples: \"0\" \"0,10,20\" \"0-5\" \"0,5-10,20\"
                            Default: 0
    --exit-after            Exit after capturing all specified frames

WORLD OPTIONS:
    --seed <N>              World generation seed (default: 42)

OTHER:
    -h, --help              Print this help message

EXAMPLES:
    # Basic viewer
    cargo run -p voxelicous-viewer

    # Use a custom seed
    cargo run -p voxelicous-viewer -- --seed 1234

    # Capture frames during orbit
    cargo run -p voxelicous-viewer -- -S -f 0,50,100,150,200 -o stream_{{}}.png --exit-after

ENVIRONMENT VARIABLES:
    RUST_LOG                Set log level (e.g., info, debug, trace)"
    );
}
