//! Application context.

use std::sync::Arc;
use std::time::Instant;

use ash::vk;
use voxelicous_gpu::swapchain::Swapchain;
use voxelicous_gpu::sync::{create_fence, create_semaphore, wait_for_fence};
use voxelicous_gpu::{GpuContext, SurfaceContext};
use winit::window::Window;

/// Application context shared across all app methods.
///
/// Provides access to the GPU context, window, swapchain, and other
/// resources needed for rendering.
pub struct AppContext {
    /// The window handle.
    pub window: Arc<Window>,
    /// GPU context with device and queues.
    pub gpu: GpuContext,
    /// Surface context for windowed rendering.
    pub surface: SurfaceContext,
    /// Current swapchain.
    pub swapchain: Swapchain,
    /// Command pool for allocating command buffers.
    pub command_pool: vk::CommandPool,
    /// Per-frame synchronization data.
    pub(crate) frames: Vec<FrameSyncData>,
    /// Per-swapchain-image render finished semaphores.
    pub(crate) render_finished_semaphores: Vec<vk::Semaphore>,
    /// Current frame index (into frames array).
    pub(crate) current_frame_index: usize,
    /// Total frames rendered.
    pub frame_count: u64,
    /// Time of last frame (for delta time calculation).
    pub(crate) last_frame_time: Instant,
    /// Whether vsync is enabled.
    pub vsync: bool,
}

/// Per-frame synchronization primitives.
pub(crate) struct FrameSyncData {
    /// Semaphore signaled when swapchain image is available.
    pub image_available: vk::Semaphore,
    /// Fence signaled when frame rendering is complete.
    pub in_flight_fence: vk::Fence,
    /// Command buffer for this frame.
    pub command_buffer: vk::CommandBuffer,
}

impl AppContext {
    /// Create a new application context.
    ///
    /// # Safety
    /// The window must have valid handles.
    pub(crate) unsafe fn new(
        window: Arc<Window>,
        gpu: GpuContext,
        vsync: bool,
    ) -> anyhow::Result<Self> {
        // Create surface
        // SAFETY: Caller guarantees window has valid handles
        let surface = unsafe { SurfaceContext::from_window(&gpu, window.as_ref())? };

        // Get window size
        let size = window.inner_size();
        let width = size.width.max(1);
        let height = size.height.max(1);

        // Create swapchain
        // SAFETY: GPU context is valid
        let swapchain = unsafe { surface.create_swapchain(&gpu, width, height, vsync, None)? };

        tracing::info!(
            "Swapchain created: {}x{} ({} images)",
            swapchain.extent.width,
            swapchain.extent.height,
            swapchain.images.len()
        );

        // Create command pool
        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(gpu.graphics_queue_family())
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        // SAFETY: Device is valid
        let command_pool = unsafe { gpu.device().create_command_pool(&pool_info, None)? };

        // Create per-frame sync data (match swapchain image count)
        let frames_in_flight = swapchain.images.len();
        let mut frames = Vec::with_capacity(frames_in_flight);
        for _ in 0..frames_in_flight {
            let alloc_info = vk::CommandBufferAllocateInfo::default()
                .command_pool(command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            // SAFETY: Device and command pool are valid
            let command_buffer = unsafe { gpu.device().allocate_command_buffers(&alloc_info)?[0] };

            frames.push(FrameSyncData {
                // SAFETY: Device is valid
                image_available: unsafe { create_semaphore(gpu.device())? },
                // SAFETY: Device is valid
                in_flight_fence: unsafe { create_fence(gpu.device(), true)? },
                command_buffer,
            });
        }

        // Create per-swapchain-image render finished semaphores
        let mut render_finished_semaphores = Vec::with_capacity(swapchain.images.len());
        for _ in 0..swapchain.images.len() {
            // SAFETY: Device is valid
            render_finished_semaphores.push(unsafe { create_semaphore(gpu.device())? });
        }

        Ok(Self {
            window,
            gpu,
            surface,
            swapchain,
            command_pool,
            frames,
            render_finished_semaphores,
            current_frame_index: 0,
            frame_count: 0,
            last_frame_time: Instant::now(),
            vsync,
        })
    }

    /// Get the current swapchain extent.
    pub fn extent(&self) -> vk::Extent2D {
        self.swapchain.extent
    }

    /// Get the swapchain width.
    pub fn width(&self) -> u32 {
        self.swapchain.extent.width
    }

    /// Get the swapchain height.
    pub fn height(&self) -> u32 {
        self.swapchain.extent.height
    }

    /// Get the aspect ratio (width / height).
    pub fn aspect_ratio(&self) -> f32 {
        self.swapchain.extent.width as f32 / self.swapchain.extent.height as f32
    }

    /// Get the number of frames in flight.
    pub fn frames_in_flight(&self) -> usize {
        self.frames.len()
    }

    /// Wait for all frame fences (all in-flight submissions) to complete.
    pub fn wait_for_all_in_flight_frames(&self, timeout_ns: u64) -> anyhow::Result<()> {
        let device = self.gpu.device();
        unsafe {
            for frame in &self.frames {
                wait_for_fence(device, frame.in_flight_fence, timeout_ns)?;
            }
        }
        Ok(())
    }

    /// Recreate the swapchain (e.g., after resize).
    ///
    /// # Safety
    /// The GPU must be idle.
    pub(crate) unsafe fn recreate_swapchain(
        &mut self,
        width: u32,
        height: u32,
    ) -> anyhow::Result<()> {
        // Destroy old swapchain
        // SAFETY: Caller guarantees GPU is idle
        unsafe {
            self.swapchain
                .destroy(self.gpu.device(), &self.surface.swapchain_loader);
        }

        // Create new swapchain
        // SAFETY: GPU context and surface are valid
        self.swapchain = unsafe {
            self.surface
                .create_swapchain(&self.gpu, width, height, self.vsync, None)?
        };

        tracing::info!(
            "Swapchain recreated: {}x{}",
            self.swapchain.extent.width,
            self.swapchain.extent.height
        );

        Ok(())
    }

    /// Cleanup all resources.
    ///
    /// # Safety
    /// The GPU must be idle and all resources must not be in use.
    pub(crate) unsafe fn cleanup(&mut self) {
        let device = self.gpu.device();

        // SAFETY: Caller guarantees GPU is idle and resources are not in use
        unsafe {
            // Destroy per-frame sync primitives
            for frame in &self.frames {
                device.destroy_semaphore(frame.image_available, None);
                device.destroy_fence(frame.in_flight_fence, None);
            }
            self.frames.clear();

            // Destroy per-swapchain-image semaphores
            for sem in &self.render_finished_semaphores {
                device.destroy_semaphore(*sem, None);
            }
            self.render_finished_semaphores.clear();

            // Destroy command pool
            device.destroy_command_pool(self.command_pool, None);

            // Destroy swapchain and surface
            self.swapchain
                .destroy(device, &self.surface.swapchain_loader);
            self.surface.destroy();
        }
    }
}
