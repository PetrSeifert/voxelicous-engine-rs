//! Modifier key flags.

use bitflags::bitflags;
use winit::keyboard::ModifiersState;

bitflags! {
    /// Modifier key flags.
    ///
    /// Represents the state of modifier keys (Shift, Ctrl, Alt, Super).
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
    pub struct Modifiers: u8 {
        /// Shift key is pressed.
        const SHIFT = 0b0000_0001;
        /// Control key is pressed.
        const CTRL  = 0b0000_0010;
        /// Alt key is pressed.
        const ALT   = 0b0000_0100;
        /// Super/Windows/Command key is pressed.
        const SUPER = 0b0000_1000;
    }
}

impl Modifiers {
    /// Returns `true` if the Shift key is pressed.
    #[inline]
    #[must_use]
    pub const fn shift(self) -> bool {
        self.contains(Self::SHIFT)
    }

    /// Returns `true` if the Control key is pressed.
    #[inline]
    #[must_use]
    pub const fn ctrl(self) -> bool {
        self.contains(Self::CTRL)
    }

    /// Returns `true` if the Alt key is pressed.
    #[inline]
    #[must_use]
    pub const fn alt(self) -> bool {
        self.contains(Self::ALT)
    }

    /// Returns `true` if the Super/Windows/Command key is pressed.
    #[inline]
    #[must_use]
    pub const fn super_key(self) -> bool {
        self.contains(Self::SUPER)
    }
}

impl From<ModifiersState> for Modifiers {
    fn from(state: ModifiersState) -> Self {
        let mut modifiers = Self::empty();
        if state.shift_key() {
            modifiers |= Self::SHIFT;
        }
        if state.control_key() {
            modifiers |= Self::CTRL;
        }
        if state.alt_key() {
            modifiers |= Self::ALT;
        }
        if state.super_key() {
            modifiers |= Self::SUPER;
        }
        modifiers
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn modifier_flags() {
        let mods = Modifiers::SHIFT | Modifiers::CTRL;
        assert!(mods.shift());
        assert!(mods.ctrl());
        assert!(!mods.alt());
        assert!(!mods.super_key());
    }

    #[test]
    fn empty_modifiers() {
        let mods = Modifiers::empty();
        assert!(!mods.shift());
        assert!(!mods.ctrl());
        assert!(!mods.alt());
        assert!(!mods.super_key());
    }
}
