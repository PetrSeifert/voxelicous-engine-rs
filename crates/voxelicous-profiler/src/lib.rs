//! Performance profiling for the Voxelicous engine.
//!
//! This crate provides instrumentation for measuring performance of various
//! engine operations like frame timing, clipmap page builds, GPU uploads, etc.
//!
//! # Feature Flags
//!
//! - `profiling`: Enable profiling instrumentation. When disabled, all profiling
//!   macros expand to no-ops for zero overhead.
//!
//! # Usage
//!
//! Initialize the profiler at application startup:
//!
//! ```ignore
//! voxelicous_profiler::init();
//! ```
//!
//! Use the profiling macros to instrument code:
//!
//! ```ignore
//! use voxelicous_profiler::{profile_scope, EventCategory};
//!
//! fn render_frame() {
//!     profile_scope!(EventCategory::Frame);
//!     // ... frame rendering code
//! }
//! ```
//!
//! At the end of each frame, call `end_frame` to flush events:
//!
//! ```ignore
//! voxelicous_profiler::end_frame(frame_number, fps, frame_time_ms);
//! ```

mod collector;
mod context;
mod events;
pub mod ipc;
mod macros;
mod ring_buffer;

// Re-export public API
pub use context::{
    end_frame, init, init_with_port, is_initialized, record, record_duration,
    record_duration_with_context, report_memory, report_queue_sizes, reset, shutdown, snapshot,
    DEFAULT_PORT,
};
pub use events::{
    CategoryStats, EventCategory, MemoryStats, ProfilerSnapshot, QueueSizes, TimingEvent,
};
pub use macros::ScopeGuard;

// Re-export protocol types for the TUI client
pub use ipc::protocol::{ClientMessage, ServerMessage, PROTOCOL_VERSION};
