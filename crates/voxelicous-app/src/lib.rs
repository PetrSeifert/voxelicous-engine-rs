//! Application framework for the Voxelicous engine.
//!
//! This crate provides a trait-based application framework that handles
//! common boilerplate like:
//! - Window creation and management
//! - GPU context initialization
//! - Swapchain creation and recreation
//! - Frame synchronization
//! - Event loop handling
//!
//! # Example
//!
//! ```no_run
//! use voxelicous_app::{VoxelApp, AppContext, FrameContext, AppConfig, run_app};
//!
//! struct MyApp {
//!     // Application state
//! }
//!
//! impl VoxelApp for MyApp {
//!     fn init(ctx: &mut AppContext) -> anyhow::Result<Self> {
//!         Ok(MyApp {})
//!     }
//!
//!     fn update(&mut self, ctx: &AppContext, dt: f32) {
//!         // Update logic
//!     }
//!
//!     fn render(&mut self, ctx: &AppContext, frame: &mut FrameContext) -> anyhow::Result<()> {
//!         // Render logic
//!         Ok(())
//!     }
//! }
//!
//! fn main() -> anyhow::Result<()> {
//!     run_app::<MyApp>(AppConfig::default())
//! }
//! ```

mod app;
mod context;
mod frame;
mod runner;

pub use app::VoxelApp;
pub use context::AppContext;
pub use frame::FrameContext;
pub use runner::{run_app, AppConfig};

// Re-export commonly used types for convenience
pub use voxelicous_gpu::{GpuContext, GpuContextBuilder};
pub use voxelicous_render::Camera;
pub use winit::event::{DeviceEvent, DeviceId, WindowEvent};
