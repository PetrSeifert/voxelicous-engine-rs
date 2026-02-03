//! Voxelicous Engine Demo Viewer
//!
//! Renders procedurally generated terrain using compute ray marching with chunk streaming.
//! Chunks are dynamically loaded and unloaded based on camera position.
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
//! ### Streaming options
//! - `--load-radius <N>`: Horizontal chunk load radius (default: 8)
//! - `--unload-radius <N>`: Horizontal chunk unload radius (default: 12)
//! - `--vertical-radius <N>`: Vertical chunk radius (default: 4)
//! - `--max-gen-per-frame <N>`: Max chunks to generate per frame (default: 4)
//! - `--seed <N>`: World generation seed (default: 42)
//!
//! ### Other
//! - `-h, --help`: Print help message
//!
//! ## Examples
//!
//! ```bash
//! # Basic viewer with default streaming
//! cargo run -p voxelicous-viewer
//!
//! # Smaller view distance for better performance
//! cargo run -p voxelicous-viewer -- --load-radius 4
//!
//! # Larger view distance
//! cargo run -p voxelicous-viewer -- --load-radius 12 --unload-radius 16
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
        AppConfig::new("Voxelicous Engine - Streaming Demo")
            .with_size(WIDTH, HEIGHT)
            .with_target_fps(TARGET_FPS),
    )
}

fn print_help() {
    eprintln!(
        "Voxelicous Engine Demo Viewer with Chunk Streaming

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

STREAMING OPTIONS:
    --load-radius <N>       Horizontal chunk load radius (default: 4)
    --unload-radius <N>     Horizontal chunk unload radius (default: 6)
    --vertical-radius <N>   Vertical chunk radius (default: 2)
    --max-gen-per-frame <N> Max chunks to generate per frame (default: 8)
    --seed <N>              World generation seed (default: 42)

NOTE: Large load radii (>8) may cause performance issues due to the O(pixels*chunks)
      complexity of the ray marching shader.

OTHER:
    -h, --help              Print this help message

EXAMPLES:
    # Basic viewer with default streaming
    cargo run -p voxelicous-viewer

    # Smaller view distance for better performance
    cargo run -p voxelicous-viewer -- --load-radius 4

    # Larger view distance
    cargo run -p voxelicous-viewer -- --load-radius 12 --unload-radius 16

    # Capture frames during orbit
    cargo run -p voxelicous-viewer -- -S -f 0,50,100,150,200 -o stream_{{}}.png --exit-after

ENVIRONMENT VARIABLES:
    RUST_LOG                Set log level (e.g., info, debug, trace)"
    );
}
