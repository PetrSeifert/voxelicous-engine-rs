//! Vulkan abstraction layer for the Voxelicous engine.
//!
//! This crate provides:
//! - Vulkan instance and device management
//! - GPU capability detection
//! - Memory allocation via gpu-allocator
//! - Command buffer management
//! - Swapchain handling

pub mod capabilities;
pub mod command;
pub mod context;
pub mod descriptors;
pub mod error;
pub mod instance;
pub mod memory;
pub mod pipeline;
pub mod surface;
pub mod swapchain;
pub mod sync;

pub use capabilities::{GpuCapabilities, GpuVendor, RayTracingCapabilities};
pub use context::{GpuContext, GpuContextBuilder};
pub use descriptors::{
    write_storage_buffer, write_storage_image, write_uniform_buffer, DescriptorPool,
    DescriptorSetLayoutBuilder,
};
pub use error::{GpuError, Result};
pub use memory::{GpuAllocator, GpuBuffer, GpuImage};
pub use pipeline::{ComputePipeline, GraphicsPipeline, GraphicsPipelineConfig};
pub use surface::{SurfaceCapabilities, SurfaceContext};
pub use sync::{create_fence, create_semaphore, FrameSync, FrameSyncManager};
