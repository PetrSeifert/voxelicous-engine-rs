//! Lock-free single-producer single-consumer ring buffer.

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::events::TimingEvent;

/// Capacity of the ring buffer (must be power of 2).
const BUFFER_SIZE: usize = 4096;

/// Lock-free SPSC ring buffer for timing events.
///
/// Uses atomic operations for thread-safe producer/consumer access.
pub struct RingBuffer {
    buffer: UnsafeCell<[TimingEvent; BUFFER_SIZE]>,
    head: AtomicUsize,
    tail: AtomicUsize,
}

// SAFETY: RingBuffer uses atomics for synchronization and is designed for SPSC use.
unsafe impl Sync for RingBuffer {}
unsafe impl Send for RingBuffer {}

impl Default for RingBuffer {
    fn default() -> Self {
        Self::new()
    }
}

impl RingBuffer {
    /// Create a new empty ring buffer.
    #[must_use]
    pub fn new() -> Self {
        Self {
            buffer: UnsafeCell::new(
                [TimingEvent::new(crate::events::EventCategory::Frame, 0); BUFFER_SIZE],
            ),
            head: AtomicUsize::new(0),
            tail: AtomicUsize::new(0),
        }
    }

    /// Push an event to the buffer.
    ///
    /// Returns `true` if successful, `false` if buffer is full.
    pub fn push(&self, event: TimingEvent) -> bool {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Acquire);

        let next_head = (head + 1) & (BUFFER_SIZE - 1);
        if next_head == tail {
            return false; // Buffer full
        }

        // SAFETY: We have exclusive write access to buffer[head] because:
        // 1. Only one producer (single-producer guarantee)
        // 2. The tail hasn't caught up to this position
        unsafe {
            (*self.buffer.get())[head] = event;
        }

        self.head.store(next_head, Ordering::Release);
        true
    }

    /// Pop an event from the buffer.
    ///
    /// Returns `None` if buffer is empty.
    pub fn pop(&self) -> Option<TimingEvent> {
        let tail = self.tail.load(Ordering::Relaxed);
        let head = self.head.load(Ordering::Acquire);

        if tail == head {
            return None; // Buffer empty
        }

        // SAFETY: We have exclusive read access to buffer[tail] because:
        // 1. Only one consumer (single-consumer guarantee)
        // 2. The head has written past this position
        let event = unsafe { (*self.buffer.get())[tail] };

        let next_tail = (tail + 1) & (BUFFER_SIZE - 1);
        self.tail.store(next_tail, Ordering::Release);

        Some(event)
    }

    /// Get the number of events in the buffer.
    #[must_use]
    pub fn len(&self) -> usize {
        let head = self.head.load(Ordering::Acquire);
        let tail = self.tail.load(Ordering::Acquire);
        (head.wrapping_sub(tail)) & (BUFFER_SIZE - 1)
    }

    /// Check if the buffer is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Drain all events from the buffer into a vector.
    pub fn drain(&self) -> Vec<TimingEvent> {
        let mut events = Vec::with_capacity(self.len());
        while let Some(event) = self.pop() {
            events.push(event);
        }
        events
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::EventCategory;

    #[test]
    fn push_pop_single() {
        let buffer = RingBuffer::new();
        let event = TimingEvent::new(EventCategory::Frame, 1000);

        assert!(buffer.push(event));
        assert_eq!(buffer.len(), 1);

        let popped = buffer.pop().unwrap();
        assert_eq!(popped.duration_ns, 1000);
        assert!(buffer.is_empty());
    }

    #[test]
    fn push_pop_multiple() {
        let buffer = RingBuffer::new();

        for i in 0..100 {
            buffer.push(TimingEvent::new(EventCategory::Frame, i));
        }

        assert_eq!(buffer.len(), 100);

        for i in 0..100 {
            let event = buffer.pop().unwrap();
            assert_eq!(event.duration_ns, i);
        }

        assert!(buffer.is_empty());
    }

    #[test]
    fn drain_all() {
        let buffer = RingBuffer::new();

        for i in 0..50 {
            buffer.push(TimingEvent::new(EventCategory::ChunkGeneration, i * 1000));
        }

        let events = buffer.drain();
        assert_eq!(events.len(), 50);
        assert!(buffer.is_empty());
    }
}
