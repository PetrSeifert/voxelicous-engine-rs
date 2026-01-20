//! GPU error types.

use ash::vk;
use thiserror::Error;

/// GPU-related errors.
#[derive(Error, Debug)]
pub enum GpuError {
    /// Vulkan error.
    #[error("Vulkan error: {0}")]
    Vulkan(#[from] vk::Result),

    /// No suitable GPU found.
    #[error("No suitable GPU found")]
    NoSuitableDevice,

    /// Required extension not supported.
    #[error("Required extension not supported: {0}")]
    ExtensionNotSupported(String),

    /// Memory allocation failed.
    #[error("Memory allocation failed: {0}")]
    AllocationFailed(String),

    /// Surface creation failed.
    #[error("Surface creation failed: {0}")]
    SurfaceCreation(String),

    /// Swapchain creation failed.
    #[error("Swapchain creation failed: {0}")]
    SwapchainCreation(String),

    /// Shader compilation failed.
    #[error("Shader compilation failed: {0}")]
    ShaderCompilation(String),

    /// Pipeline creation failed.
    #[error("Pipeline creation failed: {0}")]
    PipelineCreation(String),

    /// Resource not found.
    #[error("Resource not found: {0}")]
    ResourceNotFound(String),

    /// Invalid state.
    #[error("Invalid state: {0}")]
    InvalidState(String),

    /// Other error.
    #[error("{0}")]
    Other(String),
}

/// Result type alias.
pub type Result<T> = std::result::Result<T, GpuError>;
