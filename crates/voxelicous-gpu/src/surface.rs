//! Surface management for windowed rendering.
//!
//! Provides abstractions for Vulkan surface creation and management,
//! hiding the raw-window-handle complexity from application code.

use crate::context::GpuContext;
use crate::error::{GpuError, Result};
use crate::swapchain::{calculate_extent, select_present_mode, select_surface_format, Swapchain};
use ash::vk;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

/// Surface context for windowed rendering.
///
/// Manages the Vulkan surface and swapchain loader for a window.
pub struct SurfaceContext {
    /// The Vulkan surface handle.
    pub surface: vk::SurfaceKHR,
    /// Surface extension loader.
    pub surface_loader: ash::khr::surface::Instance,
    /// Swapchain extension loader.
    pub swapchain_loader: ash::khr::swapchain::Device,
    /// The Vulkan entry point (kept alive for surface_loader lifetime).
    #[allow(dead_code)]
    entry: ash::Entry,
}

impl SurfaceContext {
    /// Create a new surface context from a window.
    ///
    /// # Safety
    /// The GPU context must be valid and the window must have valid handles.
    pub unsafe fn from_window<W>(gpu: &GpuContext, window: &W) -> Result<Self>
    where
        W: HasDisplayHandle + HasWindowHandle,
    {
        let entry = ash::Entry::load()
            .map_err(|e| GpuError::Other(format!("Failed to load Vulkan entry: {e}")))?;

        let display = window
            .display_handle()
            .map_err(|e| GpuError::SurfaceCreation(format!("Failed to get display handle: {e}")))?;
        let window_handle = window
            .window_handle()
            .map_err(|e| GpuError::SurfaceCreation(format!("Failed to get window handle: {e}")))?;

        let surface = ash_window::create_surface(
            &entry,
            gpu.instance(),
            display.as_raw(),
            window_handle.as_raw(),
            None,
        )
        .map_err(|e| GpuError::SurfaceCreation(e.to_string()))?;

        let surface_loader = ash::khr::surface::Instance::new(&entry, gpu.instance());
        let swapchain_loader = ash::khr::swapchain::Device::new(gpu.instance(), gpu.device());

        Ok(Self {
            surface,
            surface_loader,
            swapchain_loader,
            entry,
        })
    }

    /// Query surface capabilities.
    pub fn capabilities(&self, gpu: &GpuContext) -> Result<SurfaceCapabilities> {
        unsafe {
            let caps = self
                .surface_loader
                .get_physical_device_surface_capabilities(gpu.physical_device(), self.surface)?;

            let formats = self
                .surface_loader
                .get_physical_device_surface_formats(gpu.physical_device(), self.surface)?;

            let present_modes = self
                .surface_loader
                .get_physical_device_surface_present_modes(gpu.physical_device(), self.surface)?;

            Ok(SurfaceCapabilities {
                capabilities: caps,
                formats,
                present_modes,
            })
        }
    }

    /// Create a swapchain for this surface.
    ///
    /// # Safety
    /// The GPU context must be valid.
    pub unsafe fn create_swapchain(
        &self,
        gpu: &GpuContext,
        width: u32,
        height: u32,
        vsync: bool,
        old_swapchain: Option<vk::SwapchainKHR>,
    ) -> Result<Swapchain> {
        let caps = self.capabilities(gpu)?;

        let surface_format = select_surface_format(&caps.formats);
        let present_mode = select_present_mode(&caps.present_modes, vsync);
        let extent = calculate_extent(&caps.capabilities, width, height);

        Swapchain::new(
            gpu.device(),
            &self.swapchain_loader,
            self.surface,
            &caps.capabilities,
            surface_format,
            present_mode,
            extent,
            old_swapchain,
            gpu.graphics_queue_family(),
        )
    }

    /// Recreate the swapchain with new dimensions.
    ///
    /// # Safety
    /// The old swapchain must not be in use.
    pub unsafe fn recreate_swapchain(
        &self,
        gpu: &GpuContext,
        old_swapchain: &mut Swapchain,
        width: u32,
        height: u32,
        vsync: bool,
    ) -> Result<Swapchain> {
        // Destroy old swapchain
        old_swapchain.destroy(gpu.device(), &self.swapchain_loader);

        // Create new swapchain
        self.create_swapchain(gpu, width, height, vsync, None)
    }

    /// Destroy the surface.
    ///
    /// # Safety
    /// The surface must not be in use.
    pub unsafe fn destroy(&self) {
        self.surface_loader.destroy_surface(self.surface, None);
    }
}

/// Surface capabilities query result.
pub struct SurfaceCapabilities {
    /// Raw surface capabilities.
    pub capabilities: vk::SurfaceCapabilitiesKHR,
    /// Supported surface formats.
    pub formats: Vec<vk::SurfaceFormatKHR>,
    /// Supported present modes.
    pub present_modes: Vec<vk::PresentModeKHR>,
}

impl SurfaceCapabilities {
    /// Get the recommended surface format.
    pub fn recommended_format(&self) -> vk::SurfaceFormatKHR {
        select_surface_format(&self.formats)
    }

    /// Get the recommended present mode.
    pub fn recommended_present_mode(&self, vsync: bool) -> vk::PresentModeKHR {
        select_present_mode(&self.present_modes, vsync)
    }
}
