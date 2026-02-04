//! Profiling macros and scope guards.

use std::time::Instant;

use crate::events::EventCategory;

/// RAII guard that records timing on drop.
pub struct ScopeGuard {
    category: EventCategory,
    start: Instant,
    context: [i32; 3],
}

impl ScopeGuard {
    /// Create a new scope guard.
    #[inline]
    #[must_use]
    pub fn new(category: EventCategory) -> Self {
        Self {
            category,
            start: Instant::now(),
            context: [0, 0, 0],
        }
    }

    /// Create a new scope guard with context.
    #[inline]
    #[must_use]
    pub fn with_context(category: EventCategory, context: [i32; 3]) -> Self {
        Self {
            category,
            start: Instant::now(),
            context,
        }
    }
}

impl Drop for ScopeGuard {
    #[inline]
    fn drop(&mut self) {
        let duration = self.start.elapsed().as_nanos() as u64;
        crate::context::record_duration_with_context(self.category, duration, self.context);
    }
}

/// Create a profiling scope that measures execution time until end of scope.
///
/// When the `profiling` feature is disabled, this macro expands to nothing.
///
/// # Examples
///
/// ```ignore
/// use voxelicous_profiler::{profile_scope, EventCategory};
///
/// fn build_page() {
///     profile_scope!(EventCategory::ClipmapPageBuild);
///     // ... clipmap page build code
/// } // timing recorded here
/// ```
///
/// With context:
/// ```ignore
/// profile_scope!(EventCategory::ClipmapPageBuild, [page_x, page_y, page_z]);
/// ```
#[cfg(feature = "profiling")]
#[macro_export]
macro_rules! profile_scope {
    ($category:expr) => {
        let _guard = $crate::ScopeGuard::new($category);
    };
    ($category:expr, [$cx:expr, $cy:expr, $cz:expr]) => {
        let _guard = $crate::ScopeGuard::with_context($category, [$cx, $cy, $cz]);
    };
}

#[cfg(not(feature = "profiling"))]
#[macro_export]
macro_rules! profile_scope {
    ($category:expr) => {};
    ($category:expr, [$cx:expr, $cy:expr, $cz:expr]) => {};
}

/// Record a duration directly (for when you already have the timing).
///
/// When the `profiling` feature is disabled, this macro expands to nothing.
///
/// # Examples
///
/// ```ignore
/// use voxelicous_profiler::{profile_duration, EventCategory};
/// use std::time::Instant;
///
/// let start = Instant::now();
/// do_work();
/// let elapsed = start.elapsed();
/// profile_duration!(EventCategory::Frame, elapsed);
/// ```
#[cfg(feature = "profiling")]
#[macro_export]
macro_rules! profile_duration {
    ($category:expr, $duration:expr) => {
        $crate::record_duration($category, $duration.as_nanos() as u64);
    };
}

#[cfg(not(feature = "profiling"))]
#[macro_export]
macro_rules! profile_duration {
    ($category:expr, $duration:expr) => {};
}

/// Report queue sizes to the profiler.
///
/// When the `profiling` feature is disabled, this macro expands to nothing.
#[cfg(feature = "profiling")]
#[macro_export]
macro_rules! report_queues {
    ($queues:expr) => {
        $crate::report_queue_sizes($queues);
    };
}

#[cfg(not(feature = "profiling"))]
#[macro_export]
macro_rules! report_queues {
    ($queues:expr) => {};
}

/// Report memory stats to the profiler.
///
/// When the `profiling` feature is disabled, this macro expands to nothing.
#[cfg(feature = "profiling")]
#[macro_export]
macro_rules! report_memory {
    ($memory:expr) => {
        $crate::report_memory($memory);
    };
}

#[cfg(not(feature = "profiling"))]
#[macro_export]
macro_rules! report_memory {
    ($memory:expr) => {};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scope_guard_measures_time() {
        let guard = ScopeGuard::new(EventCategory::Frame);
        std::thread::sleep(std::time::Duration::from_millis(1));
        drop(guard);
        // Just verify it doesn't panic - actual timing goes to global collector
    }
}
