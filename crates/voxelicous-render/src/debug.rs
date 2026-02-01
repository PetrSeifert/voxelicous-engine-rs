//! Debug rendering modes for visualization and profiling.
//!
//! Provides toggleable debug visualization modes for ray tracing, including
//! heatmaps and SVO visualization, controlled by hotkeys.

/// Debug visualization mode for rendering.
///
/// Press F3 to cycle through modes in the viewer.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DebugMode {
    /// Normal rendering (default).
    #[default]
    None = 0,
    /// Heatmap showing computational cost per pixel (traversal steps).
    TraversalSteps = 1,
    /// Heatmap showing SVO depth reached.
    NodeDepth = 2,
    /// Heatmap showing distance from camera.
    Distance = 3,
    /// Surface normals as RGB.
    Normals = 4,
    /// Red wireframe overlay on chunk edges.
    ChunkBoundaries = 5,
}

impl DebugMode {
    /// Cycle to the next debug mode.
    #[must_use]
    pub fn next(self) -> Self {
        match self {
            Self::None => Self::TraversalSteps,
            Self::TraversalSteps => Self::NodeDepth,
            Self::NodeDepth => Self::Distance,
            Self::Distance => Self::Normals,
            Self::Normals => Self::ChunkBoundaries,
            Self::ChunkBoundaries => Self::None,
        }
    }

    /// Get the mode as a u32 for shader push constants.
    #[must_use]
    pub const fn as_u32(self) -> u32 {
        self as u32
    }
}
