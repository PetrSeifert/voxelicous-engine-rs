//! Vulkan instance creation.

use crate::error::{GpuError, Result};
use ash::vk;
use std::ffi::{CStr, CString};

/// Required instance extensions for the engine.
pub fn required_instance_extensions() -> Vec<&'static CStr> {
    let extensions = vec![
        ash::khr::surface::NAME,
        #[cfg(target_os = "windows")]
        ash::khr::win32_surface::NAME,
        #[cfg(target_os = "linux")]
        ash::khr::xlib_surface::NAME,
        #[cfg(target_os = "linux")]
        ash::khr::wayland_surface::NAME,
        #[cfg(target_os = "macos")]
        ash::ext::metal_surface::NAME,
        #[cfg(target_os = "macos")]
        ash::khr::portability_enumeration::NAME,
    ];

    extensions
}

/// Validation layers to enable in debug builds.
pub fn validation_layers() -> Vec<&'static CStr> {
    vec![
        // Standard validation layer
        c"VK_LAYER_KHRONOS_validation",
    ]
}

/// Create a Vulkan instance.
///
/// # Safety
/// The entry must be a valid Vulkan entry point.
pub unsafe fn create_instance(
    entry: &ash::Entry,
    app_name: &str,
    enable_validation: bool,
) -> Result<ash::Instance> {
    let app_name = CString::new(app_name).unwrap();
    let engine_name = CString::new("Voxelicous").unwrap();

    let app_info = vk::ApplicationInfo::default()
        .application_name(&app_name)
        .application_version(vk::make_api_version(0, 0, 1, 0))
        .engine_name(&engine_name)
        .engine_version(vk::make_api_version(0, 0, 1, 0))
        .api_version(vk::API_VERSION_1_3);

    // Collect extension names
    let extension_names: Vec<*const i8> = required_instance_extensions()
        .iter()
        .map(|ext| ext.as_ptr())
        .collect();

    // Collect layer names
    let layers = if enable_validation {
        validation_layers()
    } else {
        vec![]
    };

    // Check that requested layers are available
    let available_layers = entry.enumerate_instance_layer_properties()?;
    for layer in &layers {
        let layer_name = layer.to_str().unwrap();
        let found = available_layers.iter().any(|props| {
            let name = CStr::from_ptr(props.layer_name.as_ptr());
            name.to_str().ok() == Some(layer_name)
        });
        if !found {
            tracing::warn!("Validation layer {} not available", layer_name);
        }
    }

    let layer_names: Vec<*const i8> = layers.iter().map(|l| l.as_ptr()).collect();

    // Required for MoltenVK on macOS
    #[cfg(target_os = "macos")]
    let create_flags = vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR;
    #[cfg(not(target_os = "macos"))]
    let create_flags = vk::InstanceCreateFlags::empty();

    let create_info = vk::InstanceCreateInfo::default()
        .application_info(&app_info)
        .enabled_extension_names(&extension_names)
        .enabled_layer_names(&layer_names)
        .flags(create_flags);

    let instance = entry.create_instance(&create_info, None)?;

    Ok(instance)
}

/// Select the best physical device.
///
/// # Safety
/// The instance must be valid.
pub unsafe fn select_physical_device(instance: &ash::Instance) -> Result<vk::PhysicalDevice> {
    let devices = instance.enumerate_physical_devices()?;

    if devices.is_empty() {
        return Err(GpuError::NoSuitableDevice);
    }

    // Score devices and pick the best
    let mut best_device = None;
    let mut best_score = 0i32;

    for device in devices {
        let score = score_physical_device(instance, device);
        if score > best_score {
            best_score = score;
            best_device = Some(device);
        }
    }

    best_device.ok_or(GpuError::NoSuitableDevice)
}

/// Score a physical device for selection.
unsafe fn score_physical_device(instance: &ash::Instance, device: vk::PhysicalDevice) -> i32 {
    let properties = instance.get_physical_device_properties(device);
    let features = instance.get_physical_device_features(device);

    // Check Vulkan 1.3 support
    let api_version = properties.api_version;
    if vk::api_version_major(api_version) < 1
        || (vk::api_version_major(api_version) == 1 && vk::api_version_minor(api_version) < 3)
    {
        return -1;
    }

    // Start scoring
    let mut score = 0;

    // Prefer discrete GPUs
    match properties.device_type {
        vk::PhysicalDeviceType::DISCRETE_GPU => score += 1000,
        vk::PhysicalDeviceType::INTEGRATED_GPU => score += 100,
        vk::PhysicalDeviceType::VIRTUAL_GPU => score += 50,
        _ => {}
    }

    // Prefer more VRAM
    let memory = instance.get_physical_device_memory_properties(device);
    let vram_mb: u64 = memory
        .memory_heaps
        .iter()
        .take(memory.memory_heap_count as usize)
        .filter(|h| h.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
        .map(|h| h.size / (1024 * 1024))
        .sum();
    score += (vram_mb / 1024) as i32; // +1 per GB

    // Prefer geometry shader support (useful for debugging)
    if features.geometry_shader == vk::TRUE {
        score += 10;
    }

    score
}
