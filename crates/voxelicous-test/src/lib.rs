//! Test harness for the Voxelicous engine.
//!
//! Provides headless rendering and visual regression testing.

pub mod harness;

pub use harness::{create_test_camera, HeadlessRenderer, VisualRegressionTest};

use thiserror::Error;

#[derive(Error, Debug)]
pub enum TestError {
    #[error("GPU error: {0}")]
    Gpu(String),
    #[error("Image comparison failed: {0}")]
    ImageComparison(String),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Image error: {0}")]
    Image(#[from] image::ImageError),
}

pub type Result<T> = std::result::Result<T, TestError>;

/// Visual regression test configuration.
#[derive(Debug, Clone)]
pub struct VisualTestConfig {
    /// Maximum allowed pixel difference (0.0-1.0).
    pub threshold: f64,
    /// Directory for baseline images.
    pub baseline_dir: String,
    /// Directory for test output images.
    pub output_dir: String,
}

impl Default for VisualTestConfig {
    fn default() -> Self {
        Self {
            threshold: 0.001,
            baseline_dir: "assets/test_data/baselines".to_string(),
            output_dir: "target/test_output".to_string(),
        }
    }
}
