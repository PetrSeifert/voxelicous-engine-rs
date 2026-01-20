//! Platform abstraction for the Voxelicous engine.
//!
//! Provides window creation and input handling via winit.

use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use thiserror::Error;
use winit::window::Window;

#[derive(Error, Debug)]
pub enum PlatformError {
    #[error("Window creation failed: {0}")]
    WindowCreation(String),
    #[error("Event loop error: {0}")]
    EventLoop(String),
}

pub type Result<T> = std::result::Result<T, PlatformError>;

/// Platform configuration.
#[derive(Debug, Clone)]
pub struct PlatformConfig {
    pub title: String,
    pub width: u32,
    pub height: u32,
    pub resizable: bool,
    pub vsync: bool,
}

impl Default for PlatformConfig {
    fn default() -> Self {
        Self {
            title: "Voxelicous Engine".to_string(),
            width: 1920,
            height: 1080,
            resizable: true,
            vsync: true,
        }
    }
}

/// Get raw handles from a window for Vulkan surface creation.
pub fn get_raw_handles(
    window: &Window,
) -> (
    raw_window_handle::RawDisplayHandle,
    raw_window_handle::RawWindowHandle,
) {
    (
        window.display_handle().unwrap().as_raw(),
        window.window_handle().unwrap().as_raw(),
    )
}
