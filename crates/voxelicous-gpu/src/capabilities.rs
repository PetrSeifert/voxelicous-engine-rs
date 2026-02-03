//! GPU capability detection.

use ash::vk;
use std::collections::HashSet;
use std::ffi::CStr;

/// GPU vendor identification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GpuVendor {
    Nvidia,
    Amd,
    Intel,
    Apple,
    Other(u32),
}

impl GpuVendor {
    /// Identify vendor from PCI vendor ID.
    pub fn from_vendor_id(id: u32) -> Self {
        match id {
            0x10DE => Self::Nvidia,
            0x1002 => Self::Amd,
            0x8086 => Self::Intel,
            0x106B => Self::Apple,
            other => Self::Other(other),
        }
    }
}

/// Detected GPU capabilities.
#[derive(Debug, Clone)]
pub struct GpuCapabilities {
    /// GPU vendor
    pub vendor: GpuVendor,
    /// Device name
    pub device_name: String,
    /// Vulkan API version
    pub api_version: u32,
    /// Driver version
    pub driver_version: u32,

    // Vulkan 1.3 core features
    /// Dynamic rendering support (VK 1.3 core)
    pub supports_dynamic_rendering: bool,
    /// Synchronization2 support (VK 1.3 core)
    pub supports_synchronization2: bool,

    // Buffer and descriptor features
    /// Buffer device address support
    pub supports_buffer_device_address: bool,
    /// Descriptor indexing support
    pub supports_descriptor_indexing: bool,
    /// Scalar block layout support
    pub supports_scalar_block_layout: bool,

    // Memory info
    /// Device-local memory in MB
    pub device_local_memory_mb: u64,
    /// Maximum memory allocation count
    pub max_memory_allocation_count: u32,

    // Compute limits
    /// Maximum compute workgroup size
    pub max_compute_workgroup_size: [u32; 3],
    /// Maximum compute workgroup invocations
    pub max_compute_workgroup_invocations: u32,
    /// Maximum compute shared memory size
    pub max_compute_shared_memory_size: u32,

    // Available extensions
    pub available_extensions: HashSet<String>,
}

impl GpuCapabilities {
    /// Query capabilities from a physical device.
    ///
    /// # Safety
    /// The instance and physical device must be valid.
    pub unsafe fn query(instance: &ash::Instance, physical_device: vk::PhysicalDevice) -> Self {
        // Get basic properties
        let properties = instance.get_physical_device_properties(physical_device);
        let memory_properties = instance.get_physical_device_memory_properties(physical_device);

        // Get available extensions
        let extensions = instance
            .enumerate_device_extension_properties(physical_device)
            .unwrap_or_default();

        let available_extensions: HashSet<String> = extensions
            .iter()
            .filter_map(|ext| {
                CStr::from_ptr(ext.extension_name.as_ptr())
                    .to_str()
                    .ok()
                    .map(String::from)
            })
            .collect();

        // Parse device info
        let vendor = GpuVendor::from_vendor_id(properties.vendor_id);
        let device_name = CStr::from_ptr(properties.device_name.as_ptr())
            .to_string_lossy()
            .into_owned();

        // Calculate device-local memory
        let device_local_memory_mb: u64 = memory_properties
            .memory_heaps
            .iter()
            .take(memory_properties.memory_heap_count as usize)
            .filter(|heap| heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
            .map(|heap| heap.size / (1024 * 1024))
            .sum();

        // Vulkan 1.3 features are core, so we check API version
        let api_version = properties.api_version;
        let has_vulkan_1_3 =
            vk::api_version_major(api_version) >= 1 && vk::api_version_minor(api_version) >= 3;

        Self {
            vendor,
            device_name,
            api_version,
            driver_version: properties.driver_version,

            supports_dynamic_rendering: has_vulkan_1_3,
            supports_synchronization2: has_vulkan_1_3,

            supports_buffer_device_address: has_vulkan_1_3
                || available_extensions.contains("VK_KHR_buffer_device_address"),
            supports_descriptor_indexing: has_vulkan_1_3
                || available_extensions.contains("VK_EXT_descriptor_indexing"),
            supports_scalar_block_layout: has_vulkan_1_3
                || available_extensions.contains("VK_EXT_scalar_block_layout"),

            device_local_memory_mb,
            max_memory_allocation_count: properties.limits.max_memory_allocation_count,

            max_compute_workgroup_size: properties.limits.max_compute_work_group_size,
            max_compute_workgroup_invocations: properties.limits.max_compute_work_group_invocations,
            max_compute_shared_memory_size: properties.limits.max_compute_shared_memory_size,

            available_extensions,
        }
    }

    /// Check if the GPU meets minimum requirements for the engine.
    pub fn meets_requirements(&self) -> bool {
        // Require Vulkan 1.3 for core features
        let api_major = vk::api_version_major(self.api_version);
        let api_minor = vk::api_version_minor(self.api_version);

        if api_major < 1 || (api_major == 1 && api_minor < 3) {
            return false;
        }

        // Require buffer device address for SVO-DAG
        if !self.supports_buffer_device_address {
            return false;
        }

        // Require at least 1GB VRAM
        if self.device_local_memory_mb < 1024 {
            return false;
        }

        true
    }

    /// Get a human-readable summary of capabilities.
    pub fn summary(&self) -> String {
        format!(
            "{} ({:?}) - Vulkan {}.{}.{} - {} MB VRAM",
            self.device_name,
            self.vendor,
            vk::api_version_major(self.api_version),
            vk::api_version_minor(self.api_version),
            vk::api_version_patch(self.api_version),
            self.device_local_memory_mb,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vendor_identification() {
        assert_eq!(GpuVendor::from_vendor_id(0x10DE), GpuVendor::Nvidia);
        assert_eq!(GpuVendor::from_vendor_id(0x1002), GpuVendor::Amd);
        assert_eq!(GpuVendor::from_vendor_id(0x8086), GpuVendor::Intel);
    }
}
