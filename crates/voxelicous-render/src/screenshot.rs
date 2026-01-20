//! Screenshot capture utilities.
//!
//! Provides reusable functionality for capturing rendered frames to image files.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use image::{ImageBuffer, Rgba};
use tracing::info;
use voxelicous_gpu::GpuError;

/// Screenshot capture configuration.
///
/// Defines which frames to capture and where to save them.
#[derive(Clone, Default)]
pub struct ScreenshotConfig {
    /// Whether screenshot capture is enabled.
    pub enabled: bool,
    /// Output path pattern (use `{}` for frame number placeholder).
    pub output_pattern: String,
    /// Frame indices to capture.
    pub frames: HashSet<u64>,
    /// Exit after capturing all specified frames.
    pub exit_after_capture: bool,
}

impl ScreenshotConfig {
    /// Create a new screenshot configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable screenshot capture with the given output pattern.
    pub fn with_output(mut self, pattern: impl Into<String>) -> Self {
        self.enabled = true;
        self.output_pattern = pattern.into();
        self
    }

    /// Add a single frame to capture.
    pub fn with_frame(mut self, frame: u64) -> Self {
        self.enabled = true;
        self.frames.insert(frame);
        self
    }

    /// Add multiple frames to capture.
    pub fn with_frames(mut self, frames: impl IntoIterator<Item = u64>) -> Self {
        self.enabled = true;
        self.frames.extend(frames);
        self
    }

    /// Set whether to exit after all captures are complete.
    pub fn with_exit_after(mut self, exit: bool) -> Self {
        self.exit_after_capture = exit;
        self
    }

    /// Get the output path for a specific frame.
    pub fn output_path(&self, frame: u64) -> PathBuf {
        PathBuf::from(self.output_pattern.replace("{}", &frame.to_string()))
    }

    /// Check if a frame should be captured.
    pub fn should_capture(&self, frame: u64) -> bool {
        self.enabled && self.frames.contains(&frame)
    }

    /// Check if all captures are complete.
    pub fn all_captured(&self, current_frame: u64) -> bool {
        if !self.enabled || self.frames.is_empty() {
            return false;
        }
        let max_frame = *self.frames.iter().max().unwrap_or(&0);
        current_frame > max_frame
    }

    /// Parse from command line arguments.
    ///
    /// Recognizes the following flags:
    /// - `-S` or `--screenshot`: Enable screenshot capture
    /// - `-o` or `--output <PATTERN>`: Output path pattern (use `{}` for frame number)
    /// - `-f` or `--frames <FRAMES>`: Frame indices to capture (e.g., "0,5,10-15")
    /// - `--exit-after`: Exit after capturing all specified frames
    pub fn from_args() -> Self {
        let args: Vec<String> = std::env::args().collect();
        Self::parse_args(&args)
    }

    /// Parse from a slice of arguments.
    pub fn parse_args(args: &[String]) -> Self {
        let mut config = Self::default();

        let mut i = 1;
        while i < args.len() {
            match args[i].as_str() {
                "-S" | "--screenshot" => {
                    config.enabled = true;
                }
                "-o" | "--output" => {
                    if i + 1 < args.len() {
                        config.output_pattern = args[i + 1].clone();
                        i += 1;
                    }
                }
                "-f" | "--frames" => {
                    if i + 1 < args.len() {
                        config.frames = parse_frame_indices(&args[i + 1]);
                        i += 1;
                    }
                }
                "--exit-after" => {
                    config.exit_after_capture = true;
                }
                _ => {}
            }
            i += 1;
        }

        // Set defaults if screenshot is enabled
        if config.enabled {
            if config.output_pattern.is_empty() {
                config.output_pattern = "screenshot_{}.png".to_string();
            }
            if config.frames.is_empty() {
                config.frames.insert(0);
            }
        }

        config
    }
}

/// Parse frame indices from a string like "0,5,10-15,20".
///
/// Supports:
/// - Single frames: "0", "5", "10"
/// - Comma-separated: "0,5,10"
/// - Ranges: "0-5" (inclusive)
/// - Mixed: "0,5-10,20"
pub fn parse_frame_indices(s: &str) -> HashSet<u64> {
    let mut frames = HashSet::new();

    for part in s.split(',') {
        let part = part.trim();
        if part.contains('-') {
            // Range: "5-10"
            let mut iter = part.splitn(2, '-');
            if let (Some(start), Some(end)) = (iter.next(), iter.next()) {
                if let (Ok(start), Ok(end)) = (start.parse::<u64>(), end.parse::<u64>()) {
                    for i in start..=end {
                        frames.insert(i);
                    }
                }
            }
        } else if let Ok(frame) = part.parse::<u64>() {
            frames.insert(frame);
        }
    }

    frames
}

/// Save RGBA pixel data to an image file.
///
/// # Arguments
/// * `data` - Raw RGBA pixel data (4 bytes per pixel)
/// * `width` - Image width in pixels
/// * `height` - Image height in pixels
/// * `path` - Output file path (format determined by extension)
///
/// # Returns
/// `Ok(())` on success, or an error if saving fails.
pub fn save_screenshot(
    data: Vec<u8>,
    width: u32,
    height: u32,
    path: impl AsRef<Path>,
) -> Result<(), ScreenshotError> {
    let path = path.as_ref();

    let image = ImageBuffer::<Rgba<u8>, _>::from_raw(width, height, data)
        .ok_or(ScreenshotError::InvalidImageData)?;

    image
        .save(path)
        .map_err(|e| ScreenshotError::SaveFailed(e.to_string()))?;

    info!("Screenshot saved: {}", path.display());
    Ok(())
}

/// Capture and save a screenshot from a pipeline's readback buffer.
///
/// This is a convenience function that reads the output from a pipeline
/// and saves it to a file.
///
/// # Arguments
/// * `read_output` - Function to read RGBA data from the pipeline
/// * `dimensions` - Function to get (width, height) of the output
/// * `path` - Output file path
pub fn capture_screenshot<F, D>(
    read_output: F,
    dimensions: D,
    path: impl AsRef<Path>,
) -> Result<(), ScreenshotError>
where
    F: FnOnce() -> Result<Vec<u8>, GpuError>,
    D: FnOnce() -> (u32, u32),
{
    let (width, height) = dimensions();
    let data = read_output().map_err(|e| ScreenshotError::ReadbackFailed(e.to_string()))?;
    save_screenshot(data, width, height, path)
}

/// Errors that can occur during screenshot capture.
#[derive(Debug)]
pub enum ScreenshotError {
    /// Failed to read pixel data from GPU.
    ReadbackFailed(String),
    /// Pixel data was invalid or wrong size.
    InvalidImageData,
    /// Failed to save image to file.
    SaveFailed(String),
}

impl std::fmt::Display for ScreenshotError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ReadbackFailed(e) => write!(f, "Failed to read screenshot data: {e}"),
            Self::InvalidImageData => write!(f, "Invalid image data"),
            Self::SaveFailed(e) => write!(f, "Failed to save screenshot: {e}"),
        }
    }
}

impl std::error::Error for ScreenshotError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_single_frame() {
        let frames = parse_frame_indices("5");
        assert_eq!(frames, HashSet::from([5]));
    }

    #[test]
    fn parse_comma_separated() {
        let frames = parse_frame_indices("0,5,10");
        assert_eq!(frames, HashSet::from([0, 5, 10]));
    }

    #[test]
    fn parse_range() {
        let frames = parse_frame_indices("3-6");
        assert_eq!(frames, HashSet::from([3, 4, 5, 6]));
    }

    #[test]
    fn parse_mixed() {
        let frames = parse_frame_indices("0,5-7,10");
        assert_eq!(frames, HashSet::from([0, 5, 6, 7, 10]));
    }

    #[test]
    fn config_output_path() {
        let config = ScreenshotConfig::new().with_output("frame_{}.png");
        assert_eq!(config.output_path(42), PathBuf::from("frame_42.png"));
    }

    #[test]
    fn config_should_capture() {
        let config = ScreenshotConfig::new().with_frames([0, 5, 10]);
        assert!(config.should_capture(0));
        assert!(config.should_capture(5));
        assert!(!config.should_capture(3));
    }
}
