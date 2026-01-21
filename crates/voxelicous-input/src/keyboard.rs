//! Keyboard input state tracking.

use hashbrown::HashMap;
use winit::event::{ElementState, KeyEvent};
use winit::keyboard::{KeyCode, PhysicalKey};

use crate::button_state::ButtonState;
use crate::modifiers::Modifiers;

/// Keyboard input state.
///
/// Tracks the state of all keyboard keys and modifier keys.
#[derive(Debug, Default)]
pub struct KeyboardState {
    /// State of individual keys by key code.
    keys: HashMap<KeyCode, ButtonState>,
    /// Current modifier key state.
    modifiers: Modifiers,
}

impl KeyboardState {
    /// Create a new keyboard state.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Process a key event.
    pub fn process_key_event(&mut self, event: &KeyEvent) {
        let PhysicalKey::Code(key_code) = event.physical_key else {
            return;
        };

        let state = self.keys.entry(key_code).or_default();
        match event.state {
            ElementState::Pressed => state.press(),
            ElementState::Released => state.release(),
        }
    }

    /// Update modifier key state.
    pub fn set_modifiers(&mut self, modifiers: Modifiers) {
        self.modifiers = modifiers;
    }

    /// Returns `true` if the key is currently pressed.
    #[must_use]
    pub fn is_pressed(&self, key: KeyCode) -> bool {
        self.keys
            .get(&key)
            .is_some_and(|s| s.is_pressed())
    }

    /// Returns `true` if the key was just pressed this frame.
    #[must_use]
    pub fn is_just_pressed(&self, key: KeyCode) -> bool {
        self.keys
            .get(&key)
            .is_some_and(|s| s.is_just_pressed())
    }

    /// Returns `true` if the key was just released this frame.
    #[must_use]
    pub fn is_just_released(&self, key: KeyCode) -> bool {
        self.keys
            .get(&key)
            .is_some_and(|s| s.is_just_released())
    }

    /// Get the current modifier key state.
    #[must_use]
    pub const fn modifiers(&self) -> Modifiers {
        self.modifiers
    }

    /// Called at end of frame to transition button states.
    pub fn end_frame(&mut self) {
        for state in self.keys.values_mut() {
            state.end_frame();
        }
    }

    /// Clear all key states.
    pub fn clear(&mut self) {
        self.keys.clear();
        self.modifiers = Modifiers::empty();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_press_and_release() {
        let mut keyboard = KeyboardState::new();

        // Key not pressed initially
        assert!(!keyboard.is_pressed(KeyCode::KeyW));
        assert!(!keyboard.is_just_pressed(KeyCode::KeyW));

        // Simulate press W by directly manipulating internal state
        keyboard.keys.entry(KeyCode::KeyW).or_default().press();
        assert!(keyboard.is_pressed(KeyCode::KeyW));
        assert!(keyboard.is_just_pressed(KeyCode::KeyW));

        // End frame
        keyboard.end_frame();
        assert!(keyboard.is_pressed(KeyCode::KeyW));
        assert!(!keyboard.is_just_pressed(KeyCode::KeyW));

        // Simulate release W
        keyboard.keys.entry(KeyCode::KeyW).or_default().release();
        assert!(!keyboard.is_pressed(KeyCode::KeyW));
        assert!(keyboard.is_just_released(KeyCode::KeyW));

        // End frame
        keyboard.end_frame();
        assert!(!keyboard.is_pressed(KeyCode::KeyW));
        assert!(!keyboard.is_just_released(KeyCode::KeyW));
    }

    #[test]
    fn modifiers() {
        let mut keyboard = KeyboardState::new();

        keyboard.set_modifiers(Modifiers::SHIFT | Modifiers::CTRL);
        assert!(keyboard.modifiers().shift());
        assert!(keyboard.modifiers().ctrl());
        assert!(!keyboard.modifiers().alt());
    }
}
