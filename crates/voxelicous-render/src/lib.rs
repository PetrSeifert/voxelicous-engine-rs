//! Ray tracing rendering pipeline for the Voxelicous engine.
//!
//! This crate provides:
//! - Compute shader ray marching (software fallback)
//! - Hardware ray tracing integration
//! - Multi-chunk world rendering
//! - Post-processing effects
//! - Camera and view management
//! - Screenshot capture utilities

pub mod camera;
pub mod ray_march;
pub mod ray_march_pipeline;
pub mod screenshot;
pub mod svo_upload;
pub mod world_render;

pub use camera::{Camera, CameraUniforms};
pub use ray_march::{Ray, RayHit, RayMarchConfig, RayMarchPushConstants};
pub use ray_march_pipeline::RayMarchPipeline;
pub use screenshot::{parse_frame_indices, save_screenshot, ScreenshotConfig, ScreenshotError};
pub use svo_upload::GpuSvoDag;
pub use world_render::{GpuChunkInfo, WorldRenderPushConstants, WorldRenderer};

use voxelicous_gpu::GpuCapabilities;

/// Render path selection based on GPU capabilities.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderPath {
    /// Compute shader ray marching (works on all Vulkan 1.3 GPUs).
    ComputeRayMarching,
    /// Hardware ray tracing (requires RTX/RDNA2+).
    HardwareRayTracing,
}

impl RenderPath {
    /// Select the best render path for the given capabilities.
    pub fn select(caps: &GpuCapabilities) -> Self {
        if caps.ray_tracing.has_hardware_rt() {
            Self::HardwareRayTracing
        } else {
            Self::ComputeRayMarching
        }
    }
}
