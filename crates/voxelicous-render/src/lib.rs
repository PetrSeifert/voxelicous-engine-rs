//! Compute ray marching rendering pipeline for the Voxelicous engine.
//!
//! This crate provides:
//! - Compute shader ray marching
//! - Multi-chunk world rendering
//! - Post-processing effects
//! - Camera and view management
//! - Screenshot capture utilities

pub mod camera;
pub mod debug;
pub mod screenshot;
pub mod svo_upload;
pub mod world_ray_march_pipeline;
pub mod world_render;

pub use camera::{Camera, CameraUniforms};
pub use debug::DebugMode;
pub use screenshot::{parse_frame_indices, save_screenshot, ScreenshotConfig, ScreenshotError};
pub use svo_upload::GpuSvoDag;
pub use world_ray_march_pipeline::WorldRayMarchPipeline;
pub use world_render::{GpuChunkInfo, WorldRenderPushConstants, WorldRenderer};
