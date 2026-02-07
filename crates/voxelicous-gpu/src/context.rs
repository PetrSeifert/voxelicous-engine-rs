//! GPU context management.

use crate::capabilities::GpuCapabilities;
use crate::error::{GpuError, Result};
use crate::instance::{create_instance, select_physical_device};
use crate::memory::GpuAllocator;
use ash::vk;
use parking_lot::Mutex;
use std::ffi::CStr;
use std::sync::Arc;

/// Main GPU context holding Vulkan resources.
pub struct GpuContext {
    // Entry must be kept alive for the lifetime of the context
    #[allow(dead_code)]
    pub(crate) entry: ash::Entry,
    pub(crate) instance: ash::Instance,
    pub(crate) physical_device: vk::PhysicalDevice,
    pub(crate) device: Arc<ash::Device>,
    pub(crate) capabilities: GpuCapabilities,
    pub(crate) allocator: Mutex<GpuAllocator>,

    // Queue families and queues
    pub(crate) graphics_queue_family: u32,
    pub(crate) compute_queue_family: u32,
    pub(crate) transfer_queue_family: u32,
    pub(crate) graphics_queue: vk::Queue,
    pub(crate) compute_queue: vk::Queue,
    pub(crate) transfer_queue: vk::Queue,
}

impl GpuContext {
    /// Get the Vulkan device handle.
    pub fn device(&self) -> &ash::Device {
        &self.device
    }

    /// Get the physical device handle.
    pub fn physical_device(&self) -> vk::PhysicalDevice {
        self.physical_device
    }

    /// Get GPU capabilities.
    pub fn capabilities(&self) -> &GpuCapabilities {
        &self.capabilities
    }

    /// Get the graphics queue.
    pub fn graphics_queue(&self) -> vk::Queue {
        self.graphics_queue
    }

    /// Get the compute queue.
    pub fn compute_queue(&self) -> vk::Queue {
        self.compute_queue
    }

    /// Get the transfer queue.
    pub fn transfer_queue(&self) -> vk::Queue {
        self.transfer_queue
    }

    /// Get the graphics queue family index.
    pub fn graphics_queue_family(&self) -> u32 {
        self.graphics_queue_family
    }

    /// Get the compute queue family index.
    pub fn compute_queue_family(&self) -> u32 {
        self.compute_queue_family
    }

    /// Get the transfer queue family index.
    pub fn transfer_queue_family(&self) -> u32 {
        self.transfer_queue_family
    }

    /// Get the Vulkan instance handle.
    pub fn instance(&self) -> &ash::Instance {
        &self.instance
    }

    /// Get access to the GPU allocator.
    pub fn allocator(&self) -> &Mutex<GpuAllocator> {
        &self.allocator
    }

    /// Wait for device to be idle.
    #[cfg_attr(
        feature = "profiling-tracy",
        tracing::instrument(level = "trace", skip_all)
    )]
    pub fn wait_idle(&self) -> Result<()> {
        unsafe {
            self.device.device_wait_idle()?;
        }
        Ok(())
    }
}

impl Drop for GpuContext {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();

            // Shutdown allocator BEFORE destroying device
            // This frees all VkDeviceMemory allocations
            self.allocator.lock().shutdown();

            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

/// Builder for creating a GPU context.
pub struct GpuContextBuilder {
    app_name: String,
    enable_validation: bool,
}

impl Default for GpuContextBuilder {
    fn default() -> Self {
        Self {
            app_name: "Voxelicous".to_string(),
            enable_validation: cfg!(debug_assertions),
        }
    }
}

impl GpuContextBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the application name.
    pub fn app_name(mut self, name: impl Into<String>) -> Self {
        self.app_name = name.into();
        self
    }

    /// Enable or disable validation layers.
    pub fn validation(mut self, enable: bool) -> Self {
        self.enable_validation = enable;
        self
    }

    /// Build the GPU context.
    pub fn build(self) -> Result<GpuContext> {
        // Load Vulkan entry point
        let entry = unsafe { ash::Entry::load() }
            .map_err(|e| GpuError::Other(format!("Failed to load Vulkan: {e}")))?;

        // Create Vulkan instance
        let instance = unsafe { create_instance(&entry, &self.app_name, self.enable_validation) }?;

        // Select best physical device
        let physical_device = unsafe { select_physical_device(&instance) }?;

        // Query capabilities
        let capabilities = unsafe { GpuCapabilities::query(&instance, physical_device) };

        // Validate requirements
        if !capabilities.meets_requirements() {
            return Err(GpuError::NoSuitableDevice);
        }

        tracing::info!("Selected GPU: {}", capabilities.summary());

        // Find queue families
        let queue_families = unsafe { find_queue_families(&instance, physical_device) }?;

        // Create logical device
        let (device, graphics_queue, compute_queue, transfer_queue) =
            unsafe { create_device(&instance, physical_device, &queue_families)? };

        let device = Arc::new(device);

        // Create GPU allocator
        let allocator = unsafe { GpuAllocator::new(&instance, device.clone(), physical_device) }?;

        Ok(GpuContext {
            entry,
            instance,
            physical_device,
            device,
            capabilities,
            allocator: Mutex::new(allocator),
            graphics_queue_family: queue_families.graphics,
            compute_queue_family: queue_families.compute,
            transfer_queue_family: queue_families.transfer,
            graphics_queue,
            compute_queue,
            transfer_queue,
        })
    }
}

/// Queue family indices.
struct QueueFamilyIndices {
    graphics: u32,
    compute: u32,
    transfer: u32,
}

/// Find queue families for graphics, compute, and transfer.
///
/// # Safety
/// The instance and physical device must be valid.
unsafe fn find_queue_families(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
) -> Result<QueueFamilyIndices> {
    let queue_families = instance.get_physical_device_queue_family_properties(physical_device);

    let mut graphics_family = None;
    let mut compute_family = None;
    let mut transfer_family = None;

    for (i, family) in queue_families.iter().enumerate() {
        let i = i as u32;

        // Look for dedicated compute queue (no graphics)
        if family.queue_flags.contains(vk::QueueFlags::COMPUTE)
            && !family.queue_flags.contains(vk::QueueFlags::GRAPHICS)
            && compute_family.is_none()
        {
            compute_family = Some(i);
        }

        // Look for dedicated transfer queue (no graphics or compute)
        if family.queue_flags.contains(vk::QueueFlags::TRANSFER)
            && !family.queue_flags.contains(vk::QueueFlags::GRAPHICS)
            && !family.queue_flags.contains(vk::QueueFlags::COMPUTE)
            && transfer_family.is_none()
        {
            transfer_family = Some(i);
        }

        // Graphics queue (also supports compute and transfer)
        if family.queue_flags.contains(vk::QueueFlags::GRAPHICS) && graphics_family.is_none() {
            graphics_family = Some(i);
        }
    }

    // Graphics queue is required
    let graphics = graphics_family.ok_or(GpuError::NoSuitableDevice)?;

    // Fall back to graphics queue for compute if no dedicated queue
    let compute = compute_family.unwrap_or(graphics);

    // Fall back to compute queue for transfer if no dedicated queue
    let transfer = transfer_family.unwrap_or(compute);

    Ok(QueueFamilyIndices {
        graphics,
        compute,
        transfer,
    })
}

/// Required device extensions.
fn required_device_extensions() -> Vec<&'static CStr> {
    let extensions = vec![
        ash::khr::swapchain::NAME,
        // Buffer device address is core in Vulkan 1.3, but some drivers still need the extension
    ];

    extensions
}

/// Create the logical device and retrieve queues.
///
/// # Safety
/// The instance and physical device must be valid.
unsafe fn create_device(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    queue_families: &QueueFamilyIndices,
) -> Result<(ash::Device, vk::Queue, vk::Queue, vk::Queue)> {
    // Collect unique queue families
    let mut unique_families = std::collections::HashSet::new();
    unique_families.insert(queue_families.graphics);
    unique_families.insert(queue_families.compute);
    unique_families.insert(queue_families.transfer);

    // Create queue create infos
    let queue_priority = 1.0_f32;
    let queue_create_infos: Vec<vk::DeviceQueueCreateInfo> = unique_families
        .iter()
        .map(|&family| {
            vk::DeviceQueueCreateInfo::default()
                .queue_family_index(family)
                .queue_priorities(std::slice::from_ref(&queue_priority))
        })
        .collect();

    // Get required extensions
    let extensions = required_device_extensions();
    let extension_names: Vec<*const i8> = extensions.iter().map(|ext| ext.as_ptr()).collect();

    // Enable Vulkan 1.3 features
    let mut vulkan_1_3_features = vk::PhysicalDeviceVulkan13Features::default()
        .dynamic_rendering(true)
        .synchronization2(true)
        .maintenance4(true);

    // Enable Vulkan 1.2 features
    let mut vulkan_1_2_features = vk::PhysicalDeviceVulkan12Features::default()
        .buffer_device_address(true)
        .descriptor_indexing(true)
        .scalar_block_layout(true)
        .runtime_descriptor_array(true)
        .shader_sampled_image_array_non_uniform_indexing(true);

    // Enable base features
    let features = vk::PhysicalDeviceFeatures::default().shader_int64(true);

    // Chain features together
    let mut features2 = vk::PhysicalDeviceFeatures2::default()
        .features(features)
        .push_next(&mut vulkan_1_3_features)
        .push_next(&mut vulkan_1_2_features);

    // Create the device
    let device_create_info = vk::DeviceCreateInfo::default()
        .queue_create_infos(&queue_create_infos)
        .enabled_extension_names(&extension_names)
        .push_next(&mut features2);

    let device = instance
        .create_device(physical_device, &device_create_info, None)
        .map_err(GpuError::from)?;

    // Get queue handles
    let graphics_queue = device.get_device_queue(queue_families.graphics, 0);
    let compute_queue = device.get_device_queue(queue_families.compute, 0);
    let transfer_queue = device.get_device_queue(queue_families.transfer, 0);

    Ok((device, graphics_queue, compute_queue, transfer_queue))
}
