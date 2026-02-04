//! Compute ray marching rendering pipeline for the Voxelicous engine.
//!
//! This crate provides:
//! - Compute shader ray marching
//! - Clipmap world rendering
//! - Post-processing effects
//! - Camera and view management
//! - Screenshot capture utilities

pub mod camera;
pub mod clipmap_ray_march_pipeline;
pub mod clipmap_render;
pub mod debug;
pub mod screenshot;

pub use camera::{Camera, CameraUniforms};
pub use clipmap_ray_march_pipeline::ClipmapRayMarchPipeline;
pub use clipmap_render::{ClipmapRenderPushConstants, ClipmapRenderer, GpuClipmapInfo};
pub use debug::DebugMode;
pub use screenshot::{parse_frame_indices, save_screenshot, ScreenshotConfig, ScreenshotError};
