//! Swapchain management.

use crate::error::{GpuError, Result};
use ash::vk;

/// Swapchain wrapper.
pub struct Swapchain {
    pub swapchain: vk::SwapchainKHR,
    pub images: Vec<vk::Image>,
    pub image_views: Vec<vk::ImageView>,
    pub format: vk::Format,
    pub extent: vk::Extent2D,
}

impl Swapchain {
    /// Create a new swapchain.
    ///
    /// # Safety
    /// All handles must be valid.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn new(
        device: &ash::Device,
        swapchain_loader: &ash::khr::swapchain::Device,
        surface: vk::SurfaceKHR,
        surface_capabilities: &vk::SurfaceCapabilitiesKHR,
        surface_format: vk::SurfaceFormatKHR,
        present_mode: vk::PresentModeKHR,
        extent: vk::Extent2D,
        old_swapchain: Option<vk::SwapchainKHR>,
        graphics_queue_family: u32,
    ) -> Result<Self> {
        // Determine image count
        let mut image_count = surface_capabilities.min_image_count + 1;
        if surface_capabilities.max_image_count > 0
            && image_count > surface_capabilities.max_image_count
        {
            image_count = surface_capabilities.max_image_count;
        }

        let queue_families = [graphics_queue_family];
        let create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(surface)
            .min_image_count(image_count)
            .image_format(surface_format.format)
            .image_color_space(surface_format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .queue_family_indices(&queue_families)
            .pre_transform(surface_capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true)
            .old_swapchain(old_swapchain.unwrap_or(vk::SwapchainKHR::null()));

        let swapchain = swapchain_loader
            .create_swapchain(&create_info, None)
            .map_err(|e| GpuError::SwapchainCreation(e.to_string()))?;

        // Get swapchain images
        let images = swapchain_loader.get_swapchain_images(swapchain)?;

        // Create image views
        let image_views: Vec<_> = images
            .iter()
            .map(|&image| {
                let view_info = vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(surface_format.format)
                    .components(vk::ComponentMapping::default())
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .base_mip_level(0)
                            .level_count(1)
                            .base_array_layer(0)
                            .layer_count(1),
                    );

                device.create_image_view(&view_info, None)
            })
            .collect::<std::result::Result<Vec<_>, _>>()?;

        Ok(Self {
            swapchain,
            images,
            image_views,
            format: surface_format.format,
            extent,
        })
    }

    /// Acquire the next image.
    ///
    /// # Safety
    /// All handles must be valid.
    #[cfg_attr(
        feature = "profiling-tracy",
        tracing::instrument(level = "trace", skip_all)
    )]
    pub unsafe fn acquire_next_image(
        &self,
        swapchain_loader: &ash::khr::swapchain::Device,
        semaphore: vk::Semaphore,
        timeout_ns: u64,
    ) -> Result<(u32, bool)> {
        let result = swapchain_loader.acquire_next_image(
            self.swapchain,
            timeout_ns,
            semaphore,
            vk::Fence::null(),
        );

        match result {
            Ok((index, suboptimal)) => Ok((index, suboptimal)),
            // OUT_OF_DATE means no image was acquired; caller must recreate the swapchain.
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                Err(GpuError::Vulkan(vk::Result::ERROR_OUT_OF_DATE_KHR))
            }
            Err(e) => Err(GpuError::from(e)),
        }
    }

    /// Present an image.
    ///
    /// # Safety
    /// All handles must be valid.
    #[cfg_attr(
        feature = "profiling-tracy",
        tracing::instrument(level = "trace", skip_all)
    )]
    pub unsafe fn present(
        &self,
        swapchain_loader: &ash::khr::swapchain::Device,
        queue: vk::Queue,
        image_index: u32,
        wait_semaphores: &[vk::Semaphore],
    ) -> Result<bool> {
        let swapchains = [self.swapchain];
        let image_indices = [image_index];

        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(wait_semaphores)
            .swapchains(&swapchains)
            .image_indices(&image_indices);

        let result = swapchain_loader.queue_present(queue, &present_info);

        match result {
            Ok(suboptimal) => Ok(suboptimal),
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => Ok(true),
            Err(e) => Err(GpuError::from(e)),
        }
    }

    /// Destroy the swapchain.
    ///
    /// # Safety
    /// All handles must be valid and swapchain must not be in use.
    pub unsafe fn destroy(
        &self,
        device: &ash::Device,
        swapchain_loader: &ash::khr::swapchain::Device,
    ) {
        for &view in &self.image_views {
            device.destroy_image_view(view, None);
        }
        swapchain_loader.destroy_swapchain(self.swapchain, None);
    }
}

/// Select the best surface format.
pub fn select_surface_format(available: &[vk::SurfaceFormatKHR]) -> vk::SurfaceFormatKHR {
    // Prefer SRGB
    for format in available {
        if format.format == vk::Format::B8G8R8A8_SRGB
            && format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
        {
            return *format;
        }
    }

    // Fall back to first available
    available[0]
}

/// Select the best present mode.
pub fn select_present_mode(available: &[vk::PresentModeKHR], vsync: bool) -> vk::PresentModeKHR {
    if vsync {
        // Prefer FIFO (vsync)
        vk::PresentModeKHR::FIFO
    } else {
        // Prefer mailbox (triple buffering without vsync)
        for &mode in available {
            if mode == vk::PresentModeKHR::MAILBOX {
                return mode;
            }
        }
        // Fall back to immediate
        for &mode in available {
            if mode == vk::PresentModeKHR::IMMEDIATE {
                return mode;
            }
        }
        // Fall back to FIFO (always supported)
        vk::PresentModeKHR::FIFO
    }
}

/// Calculate swapchain extent.
pub fn calculate_extent(
    capabilities: &vk::SurfaceCapabilitiesKHR,
    desired_width: u32,
    desired_height: u32,
) -> vk::Extent2D {
    if capabilities.current_extent.width != u32::MAX {
        capabilities.current_extent
    } else {
        vk::Extent2D {
            width: desired_width.clamp(
                capabilities.min_image_extent.width,
                capabilities.max_image_extent.width,
            ),
            height: desired_height.clamp(
                capabilities.min_image_extent.height,
                capabilities.max_image_extent.height,
            ),
        }
    }
}
