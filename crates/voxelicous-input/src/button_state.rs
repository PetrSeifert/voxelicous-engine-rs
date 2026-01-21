//! Button state tracking for keys and mouse buttons.

/// State of a button (key or mouse button).
///
/// Tracks press/release transitions for handling single-frame events:
/// - `JustPressed`: Button was pressed this frame
/// - `Pressed`: Button is being held down
/// - `JustReleased`: Button was released this frame
/// - `Released`: Button is not pressed
///
/// State transitions:
/// ```text
/// Released ─press()─> JustPressed ─end_frame()─> Pressed
///     ^                                             │
///     │                                          release()
///     │                                             v
///     └────────────end_frame()───────────── JustReleased
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ButtonState {
    /// Button was pressed this frame.
    JustPressed,
    /// Button is being held down.
    Pressed,
    /// Button was released this frame.
    JustReleased,
    /// Button is not pressed.
    #[default]
    Released,
}

impl ButtonState {
    /// Returns `true` if the button is currently pressed (including just pressed).
    #[inline]
    #[must_use]
    pub const fn is_pressed(self) -> bool {
        matches!(self, Self::JustPressed | Self::Pressed)
    }

    /// Returns `true` if the button was just pressed this frame.
    #[inline]
    #[must_use]
    pub const fn is_just_pressed(self) -> bool {
        matches!(self, Self::JustPressed)
    }

    /// Returns `true` if the button was just released this frame.
    #[inline]
    #[must_use]
    pub const fn is_just_released(self) -> bool {
        matches!(self, Self::JustReleased)
    }

    /// Returns `true` if the button is not pressed.
    #[inline]
    #[must_use]
    pub const fn is_released(self) -> bool {
        matches!(self, Self::JustReleased | Self::Released)
    }

    /// Transition to pressed state.
    #[inline]
    pub fn press(&mut self) {
        if !self.is_pressed() {
            *self = Self::JustPressed;
        }
    }

    /// Transition to released state.
    #[inline]
    pub fn release(&mut self) {
        if self.is_pressed() {
            *self = Self::JustReleased;
        }
    }

    /// Called at end of frame to transition just-pressed/released states.
    #[inline]
    pub fn end_frame(&mut self) {
        match *self {
            Self::JustPressed => *self = Self::Pressed,
            Self::JustReleased => *self = Self::Released,
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn button_state_transitions() {
        let mut state = ButtonState::Released;
        assert!(!state.is_pressed());
        assert!(!state.is_just_pressed());
        assert!(state.is_released());

        // Press the button
        state.press();
        assert!(state.is_pressed());
        assert!(state.is_just_pressed());
        assert!(!state.is_released());

        // End frame - transition to held
        state.end_frame();
        assert!(state.is_pressed());
        assert!(!state.is_just_pressed());

        // Release the button
        state.release();
        assert!(!state.is_pressed());
        assert!(state.is_just_released());
        assert!(state.is_released());

        // End frame - transition to released
        state.end_frame();
        assert!(!state.is_pressed());
        assert!(!state.is_just_released());
        assert!(state.is_released());
    }

    #[test]
    fn double_press_ignored() {
        let mut state = ButtonState::JustPressed;
        state.press();
        assert_eq!(state, ButtonState::JustPressed);

        state.end_frame();
        state.press();
        assert_eq!(state, ButtonState::Pressed);
    }

    #[test]
    fn double_release_ignored() {
        let mut state = ButtonState::Released;
        state.release();
        assert_eq!(state, ButtonState::Released);
    }
}
