//! Mouse input state tracking.

use glam::Vec2;
use winit::event::{ElementState, MouseButton as WinitMouseButton, MouseScrollDelta};

use crate::button_state::ButtonState;

/// Mouse button identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MouseButton {
    Left,
    Right,
    Middle,
    Back,
    Forward,
}

impl TryFrom<WinitMouseButton> for MouseButton {
    type Error = ();

    fn try_from(button: WinitMouseButton) -> Result<Self, Self::Error> {
        match button {
            WinitMouseButton::Left => Ok(Self::Left),
            WinitMouseButton::Right => Ok(Self::Right),
            WinitMouseButton::Middle => Ok(Self::Middle),
            WinitMouseButton::Back => Ok(Self::Back),
            WinitMouseButton::Forward => Ok(Self::Forward),
            WinitMouseButton::Other(_) => Err(()),
        }
    }
}

/// Cursor mode for the window.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CursorMode {
    /// Normal cursor behavior.
    #[default]
    Normal,
    /// Cursor is confined to the window.
    Confined,
    /// Cursor is locked and hidden (for FPS-style controls).
    Locked,
}

/// Mouse input state.
///
/// Tracks mouse position, movement, button states, and scroll wheel.
#[derive(Debug, Default)]
pub struct MouseState {
    /// Current cursor position in window coordinates.
    position: Vec2,
    /// Mouse movement delta since last frame.
    delta: Vec2,
    /// Raw mouse motion delta (for locked cursor mode).
    raw_delta: Vec2,
    /// Scroll wheel delta (x = horizontal, y = vertical).
    scroll_delta: Vec2,
    /// Button states.
    buttons: [ButtonState; 5],
    /// Current cursor mode.
    cursor_mode: CursorMode,
}

impl MouseState {
    /// Create a new mouse state.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Process cursor position update.
    #[allow(clippy::cast_possible_truncation)]
    pub fn set_position(&mut self, x: f64, y: f64) {
        let new_pos = Vec2::new(x as f32, y as f32);
        self.delta = new_pos - self.position;
        self.position = new_pos;
    }

    /// Process raw mouse motion (device motion, ignoring cursor position).
    #[allow(clippy::cast_possible_truncation)]
    pub fn add_raw_motion(&mut self, delta_x: f64, delta_y: f64) {
        self.raw_delta.x += delta_x as f32;
        self.raw_delta.y += delta_y as f32;
    }

    /// Process mouse button event.
    pub fn process_button(&mut self, button: WinitMouseButton, state: ElementState) {
        let Ok(button) = MouseButton::try_from(button) else {
            return;
        };

        let button_state = &mut self.buttons[button as usize];
        match state {
            ElementState::Pressed => button_state.press(),
            ElementState::Released => button_state.release(),
        }
    }

    /// Process scroll wheel event.
    #[allow(clippy::cast_possible_truncation)]
    pub fn process_scroll(&mut self, delta: MouseScrollDelta) {
        match delta {
            MouseScrollDelta::LineDelta(x, y) => {
                self.scroll_delta.x += x;
                self.scroll_delta.y += y;
            }
            MouseScrollDelta::PixelDelta(pos) => {
                // Convert pixel delta to line delta (approximate)
                self.scroll_delta.x += pos.x as f32 / 100.0;
                self.scroll_delta.y += pos.y as f32 / 100.0;
            }
        }
    }

    /// Get current cursor position in window coordinates.
    #[must_use]
    pub const fn position(&self) -> Vec2 {
        self.position
    }

    /// Get mouse movement delta since last frame.
    #[must_use]
    pub const fn delta(&self) -> Vec2 {
        self.delta
    }

    /// Get raw mouse motion delta (useful for locked cursor mode).
    #[must_use]
    pub const fn raw_delta(&self) -> Vec2 {
        self.raw_delta
    }

    /// Get scroll wheel delta (x = horizontal, y = vertical).
    #[must_use]
    pub const fn scroll_delta(&self) -> Vec2 {
        self.scroll_delta
    }

    /// Returns `true` if the button is currently pressed.
    #[must_use]
    pub const fn is_pressed(&self, button: MouseButton) -> bool {
        self.buttons[button as usize].is_pressed()
    }

    /// Returns `true` if the button was just pressed this frame.
    #[must_use]
    pub const fn is_just_pressed(&self, button: MouseButton) -> bool {
        self.buttons[button as usize].is_just_pressed()
    }

    /// Returns `true` if the button was just released this frame.
    #[must_use]
    pub const fn is_just_released(&self, button: MouseButton) -> bool {
        self.buttons[button as usize].is_just_released()
    }

    /// Get the current cursor mode.
    #[must_use]
    pub const fn cursor_mode(&self) -> CursorMode {
        self.cursor_mode
    }

    /// Set the cursor mode.
    pub fn set_cursor_mode(&mut self, mode: CursorMode) {
        self.cursor_mode = mode;
    }

    /// Called at end of frame to transition button states and reset deltas.
    pub fn end_frame(&mut self) {
        for button in &mut self.buttons {
            button.end_frame();
        }
        self.delta = Vec2::ZERO;
        self.raw_delta = Vec2::ZERO;
        self.scroll_delta = Vec2::ZERO;
    }

    /// Clear all mouse state.
    pub fn clear(&mut self) {
        self.position = Vec2::ZERO;
        self.delta = Vec2::ZERO;
        self.raw_delta = Vec2::ZERO;
        self.scroll_delta = Vec2::ZERO;
        self.buttons = [ButtonState::Released; 5];
        self.cursor_mode = CursorMode::Normal;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use winit::event::ElementState;

    #[test]
    fn mouse_position() {
        let mut mouse = MouseState::new();

        mouse.set_position(100.0, 200.0);
        assert_eq!(mouse.position(), Vec2::new(100.0, 200.0));

        mouse.set_position(150.0, 220.0);
        assert_eq!(mouse.position(), Vec2::new(150.0, 220.0));
        assert_eq!(mouse.delta(), Vec2::new(50.0, 20.0));
    }

    #[test]
    fn mouse_buttons() {
        let mut mouse = MouseState::new();

        assert!(!mouse.is_pressed(MouseButton::Left));

        mouse.process_button(WinitMouseButton::Left, ElementState::Pressed);
        assert!(mouse.is_pressed(MouseButton::Left));
        assert!(mouse.is_just_pressed(MouseButton::Left));

        mouse.end_frame();
        assert!(mouse.is_pressed(MouseButton::Left));
        assert!(!mouse.is_just_pressed(MouseButton::Left));

        mouse.process_button(WinitMouseButton::Left, ElementState::Released);
        assert!(!mouse.is_pressed(MouseButton::Left));
        assert!(mouse.is_just_released(MouseButton::Left));
    }

    #[test]
    fn scroll_delta() {
        let mut mouse = MouseState::new();

        mouse.process_scroll(MouseScrollDelta::LineDelta(0.0, 1.0));
        assert_eq!(mouse.scroll_delta(), Vec2::new(0.0, 1.0));

        mouse.process_scroll(MouseScrollDelta::LineDelta(0.5, 0.5));
        assert_eq!(mouse.scroll_delta(), Vec2::new(0.5, 1.5));

        mouse.end_frame();
        assert_eq!(mouse.scroll_delta(), Vec2::ZERO);
    }

    #[test]
    fn raw_motion() {
        let mut mouse = MouseState::new();

        mouse.add_raw_motion(10.0, 20.0);
        assert_eq!(mouse.raw_delta(), Vec2::new(10.0, 20.0));

        mouse.add_raw_motion(5.0, 5.0);
        assert_eq!(mouse.raw_delta(), Vec2::new(15.0, 25.0));

        mouse.end_frame();
        assert_eq!(mouse.raw_delta(), Vec2::ZERO);
    }
}
