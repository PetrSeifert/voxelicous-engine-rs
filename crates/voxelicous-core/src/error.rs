//! Error types for the engine.

use thiserror::Error;

/// Engine-wide error type.
#[derive(Error, Debug)]
pub enum Error {
    /// I/O error
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(String),

    /// Invalid data error
    #[error("Invalid data: {0}")]
    InvalidData(String),

    /// Resource not found
    #[error("Resource not found: {0}")]
    NotFound(String),

    /// Out of bounds access
    #[error("Out of bounds: {0}")]
    OutOfBounds(String),

    /// GPU error
    #[error("GPU error: {0}")]
    Gpu(String),

    /// Platform error
    #[error("Platform error: {0}")]
    Platform(String),
}

/// Result type alias using our Error type.
pub type Result<T> = std::result::Result<T, Error>;
