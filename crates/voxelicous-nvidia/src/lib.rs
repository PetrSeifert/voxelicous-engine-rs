//! Nvidia RTX optimizations for the Voxelicous engine.
//!
//! This crate provides hardware ray tracing acceleration using:
//! - `VK_KHR_ray_tracing_pipeline`
//! - `VK_KHR_acceleration_structure`
//! - DLSS integration via Streamline SDK (future)
//!
//! # Feature Flags
//!
//! - `nvidia` - Enable RTX acceleration features (requires compatible GPU)
//!
//! # Example
//!
//! ```ignore
//! use voxelicous_nvidia::RayTracePipeline;
//!
//! // Create ray tracing pipeline (requires nvidia feature)
//! let pipeline = unsafe {
//!     RayTracePipeline::new(
//!         device, instance, physical_device, allocator,
//!         1920, 1080, 5, // width, height, octree_depth
//!     )?
//! };
//! ```

#[cfg(feature = "nvidia")]
pub mod acceleration;
#[cfg(feature = "nvidia")]
pub mod ray_trace;
#[cfg(feature = "nvidia")]
pub mod sbt;

// Re-exports for convenience
#[cfg(feature = "nvidia")]
pub use acceleration::{AabbPositions, ProceduralBlas, SceneAccelerationStructure, Tlas};
#[cfg(feature = "nvidia")]
pub use ray_trace::{RayTracePipeline, RayTracePushConstants};
#[cfg(feature = "nvidia")]
pub use sbt::ShaderBindingTable;

/// Check if Nvidia features are available at runtime.
pub fn is_available() -> bool {
    cfg!(feature = "nvidia")
}
