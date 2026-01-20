//! Unified rendering pipeline abstraction.
//!
//! Provides a common interface for both compute ray marching and hardware ray tracing
//! pipelines, automatically selecting the best path based on GPU capabilities.

use ash::vk;
use tracing::info;
#[cfg(not(feature = "nvidia"))]
use tracing::warn;
use voxelicous_gpu::memory::{GpuAllocator, GpuImage};
use voxelicous_gpu::{GpuContext, GpuError};
use voxelicous_render::camera::CameraUniforms;
use voxelicous_render::ray_march_pipeline::RayMarchPipeline;
use voxelicous_render::svo_upload::GpuSvoDag;
use voxelicous_render::RenderPath;

#[cfg(feature = "nvidia")]
use voxelicous_nvidia::RayTracePipeline;

/// Unified rendering pipeline abstraction.
///
/// Wraps either the compute ray marching pipeline or the hardware RT pipeline,
/// providing a common interface for rendering.
pub enum UnifiedPipeline {
    /// Compute shader ray marching (works on all Vulkan 1.3 GPUs).
    Compute(RayMarchPipeline),
    /// Hardware ray tracing (requires RTX/RDNA2+).
    #[cfg(feature = "nvidia")]
    HardwareRT(RayTracePipeline),
}

impl UnifiedPipeline {
    /// Create the appropriate pipeline based on GPU capabilities.
    ///
    /// # Safety
    /// All Vulkan handles must be valid.
    #[allow(unused_variables)]
    pub unsafe fn new(
        gpu: &GpuContext,
        allocator: &mut GpuAllocator,
        width: u32,
        height: u32,
        octree_depth: u32,
    ) -> anyhow::Result<(Self, RenderPath)> {
        let render_path = Self::select_render_path(gpu);

        let pipeline = match render_path {
            #[cfg(feature = "nvidia")]
            RenderPath::HardwareRayTracing => {
                info!("Creating hardware ray tracing pipeline (RTX)");
                // SAFETY: Caller guarantees all handles are valid
                let rt_pipeline = unsafe {
                    RayTracePipeline::new(
                        gpu.device(),
                        gpu.instance(),
                        gpu.physical_device(),
                        allocator,
                        width,
                        height,
                        octree_depth,
                    )?
                };
                Self::HardwareRT(rt_pipeline)
            }
            #[cfg(not(feature = "nvidia"))]
            RenderPath::HardwareRayTracing => {
                warn!(
                    "Hardware RT requested but nvidia feature not enabled, falling back to compute"
                );
                // SAFETY: Caller guarantees all handles are valid
                let compute_pipeline =
                    unsafe { RayMarchPipeline::new(gpu.device(), allocator, width, height)? };
                Self::Compute(compute_pipeline)
            }
            RenderPath::ComputeRayMarching => {
                info!("Creating compute ray marching pipeline");
                // SAFETY: Caller guarantees all handles are valid
                let compute_pipeline =
                    unsafe { RayMarchPipeline::new(gpu.device(), allocator, width, height)? };
                Self::Compute(compute_pipeline)
            }
        };

        Ok((pipeline, render_path))
    }

    /// Select the render path, respecting environment variable override.
    pub fn select_render_path(gpu: &GpuContext) -> RenderPath {
        // Allow forcing compute path via environment variable
        if std::env::var("VOXELICOUS_FORCE_COMPUTE").is_ok() {
            info!("VOXELICOUS_FORCE_COMPUTE set, using compute ray marching");
            return RenderPath::ComputeRayMarching;
        }

        RenderPath::select(gpu.capabilities())
    }

    /// Build acceleration structures (only needed for hardware RT).
    ///
    /// # Safety
    /// Command buffer must be in recording state.
    #[cfg(feature = "nvidia")]
    pub unsafe fn build_acceleration_structures(&self, gpu: &GpuContext, cmd: vk::CommandBuffer) {
        if let Self::HardwareRT(rt) = self {
            // SAFETY: Caller guarantees command buffer is in recording state
            unsafe {
                rt.build_acceleration_structures(gpu.device(), gpu.instance(), cmd);
            }
        }
    }

    /// Build acceleration structures (no-op when nvidia feature is disabled).
    #[cfg(not(feature = "nvidia"))]
    pub unsafe fn build_acceleration_structures(&self, _gpu: &GpuContext, _cmd: vk::CommandBuffer) {
        // No-op for compute path
    }

    /// Record rendering commands.
    ///
    /// # Safety
    /// Command buffer must be in recording state.
    pub unsafe fn record(
        &self,
        gpu: &GpuContext,
        cmd: vk::CommandBuffer,
        camera: &CameraUniforms,
        svo: &GpuSvoDag,
        max_steps: u32,
    ) -> anyhow::Result<()> {
        match self {
            Self::Compute(pipeline) => {
                // SAFETY: Caller guarantees command buffer is in recording state
                unsafe {
                    pipeline.record(gpu.device(), cmd, camera, svo, max_steps)?;
                }
            }
            #[cfg(feature = "nvidia")]
            Self::HardwareRT(pipeline) => {
                // SAFETY: Caller guarantees command buffer is in recording state
                unsafe {
                    pipeline.record(gpu.device(), gpu.instance(), cmd, camera, svo, max_steps)?;
                }
            }
        }
        Ok(())
    }

    /// Get output image dimensions.
    pub fn dimensions(&self) -> (u32, u32) {
        match self {
            Self::Compute(p) => p.dimensions(),
            #[cfg(feature = "nvidia")]
            Self::HardwareRT(p) => p.dimensions(),
        }
    }

    /// Get the output image.
    pub fn output_image(&self) -> &GpuImage {
        match self {
            Self::Compute(p) => p.output_image(),
            #[cfg(feature = "nvidia")]
            Self::HardwareRT(p) => p.output_image(),
        }
    }

    /// Destroy the pipeline.
    ///
    /// # Safety
    /// All resources must not be in use.
    pub unsafe fn destroy(
        self,
        gpu: &GpuContext,
        allocator: &mut GpuAllocator,
    ) -> anyhow::Result<()> {
        match self {
            Self::Compute(p) => {
                // SAFETY: Caller guarantees resources are not in use
                unsafe {
                    p.destroy(gpu.device(), allocator)?;
                }
            }
            #[cfg(feature = "nvidia")]
            Self::HardwareRT(p) => {
                // SAFETY: Caller guarantees resources are not in use
                unsafe {
                    p.destroy(gpu.device(), gpu.instance(), allocator)?;
                }
            }
        }
        Ok(())
    }

    /// Record commands to copy the output image to the readback buffer.
    ///
    /// This variant assumes the output image is already in `TRANSFER_SRC_OPTIMAL` layout
    /// (e.g., after a blit operation).
    ///
    /// # Safety
    /// Command buffer must be in recording state.
    /// Output image must be in `TRANSFER_SRC_OPTIMAL` layout.
    pub unsafe fn record_readback_from_transfer_src(
        &self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
    ) {
        // SAFETY: Caller guarantees command buffer is in recording state
        // and output image is in TRANSFER_SRC_OPTIMAL layout
        unsafe {
            match self {
                Self::Compute(p) => p.record_readback_from_transfer_src(device, cmd),
                #[cfg(feature = "nvidia")]
                Self::HardwareRT(p) => p.record_readback_from_transfer_src(device, cmd),
            }
        }
    }

    /// Read the rendered image from the readback buffer.
    pub fn read_output(&self) -> Result<Vec<u8>, GpuError> {
        match self {
            Self::Compute(p) => p.read_output(),
            #[cfg(feature = "nvidia")]
            Self::HardwareRT(p) => p.read_output(),
        }
    }

    /// Get the pipeline stage flags appropriate for this render path.
    pub fn output_stage(&self) -> vk::PipelineStageFlags2 {
        match self {
            Self::Compute(_) => vk::PipelineStageFlags2::COMPUTE_SHADER,
            #[cfg(feature = "nvidia")]
            Self::HardwareRT(_) => vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
        }
    }

    /// Check if this is a hardware RT pipeline.
    #[cfg(feature = "nvidia")]
    pub fn is_hardware_rt(&self) -> bool {
        matches!(self, Self::HardwareRT(_))
    }

    /// Check if this is a hardware RT pipeline (always false without nvidia feature).
    #[cfg(not(feature = "nvidia"))]
    pub fn is_hardware_rt(&self) -> bool {
        false
    }
}
