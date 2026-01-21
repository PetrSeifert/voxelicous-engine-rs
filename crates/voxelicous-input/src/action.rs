//! Action mapping system for input handling.

use hashbrown::HashMap;
use winit::keyboard::KeyCode;

use crate::button_state::ButtonState;
use crate::keyboard::KeyboardState;
use crate::modifiers::Modifiers;
use crate::mouse::{MouseButton, MouseState};

/// An input that can be bound to an action.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InputBinding {
    /// A keyboard key.
    Key(KeyCode),
    /// A keyboard key with required modifiers.
    KeyWithModifiers(KeyCode, Modifiers),
    /// A mouse button.
    Mouse(MouseButton),
}

impl InputBinding {
    /// Create a key binding.
    #[must_use]
    pub const fn key(key: KeyCode) -> Self {
        Self::Key(key)
    }

    /// Create a key binding with modifiers.
    #[must_use]
    pub const fn key_with_modifiers(key: KeyCode, modifiers: Modifiers) -> Self {
        Self::KeyWithModifiers(key, modifiers)
    }

    /// Create a mouse button binding.
    #[must_use]
    pub const fn mouse(button: MouseButton) -> Self {
        Self::Mouse(button)
    }
}

impl From<KeyCode> for InputBinding {
    fn from(key: KeyCode) -> Self {
        Self::Key(key)
    }
}

impl From<MouseButton> for InputBinding {
    fn from(button: MouseButton) -> Self {
        Self::Mouse(button)
    }
}

/// An action that can be triggered by input bindings.
#[derive(Debug)]
struct Action {
    /// Input bindings for this action.
    bindings: Vec<InputBinding>,
    /// Computed state based on bindings.
    state: ButtonState,
}

impl Action {
    const fn new() -> Self {
        Self {
            bindings: Vec::new(),
            state: ButtonState::Released,
        }
    }

    fn add_binding(&mut self, binding: InputBinding) {
        if !self.bindings.contains(&binding) {
            self.bindings.push(binding);
        }
    }

    fn remove_binding(&mut self, binding: InputBinding) {
        self.bindings.retain(|b| *b != binding);
    }

    fn clear_bindings(&mut self) {
        self.bindings.clear();
    }
}

/// Action mapping system.
///
/// Maps named actions to input bindings, allowing multiple inputs per action.
#[derive(Debug, Default)]
pub struct ActionMap {
    /// Actions by name.
    actions: HashMap<String, Action>,
}

impl ActionMap {
    /// Create a new action map.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a new action map with a builder pattern.
    #[must_use]
    pub fn builder() -> ActionMapBuilder {
        ActionMapBuilder::new()
    }

    /// Add a binding to an action, creating the action if it doesn't exist.
    pub fn bind(&mut self, action: impl Into<String>, binding: impl Into<InputBinding>) {
        let action_name = action.into();
        let binding = binding.into();
        self.actions
            .entry(action_name)
            .or_insert_with(Action::new)
            .add_binding(binding);
    }

    /// Remove a binding from an action.
    pub fn unbind(&mut self, action: &str, binding: InputBinding) {
        if let Some(action) = self.actions.get_mut(action) {
            action.remove_binding(binding);
        }
    }

    /// Clear all bindings for an action.
    pub fn clear_bindings(&mut self, action: &str) {
        if let Some(action) = self.actions.get_mut(action) {
            action.clear_bindings();
        }
    }

    /// Get the bindings for an action.
    #[must_use]
    pub fn get_bindings(&self, action: &str) -> Option<&[InputBinding]> {
        self.actions.get(action).map(|a| a.bindings.as_slice())
    }

    /// Update action states based on keyboard and mouse state.
    pub fn update(&mut self, keyboard: &KeyboardState, mouse: &MouseState) {
        for action in self.actions.values_mut() {
            // Check if any binding is pressed
            let any_pressed = action.bindings.iter().any(|binding| match binding {
                InputBinding::Key(key) => keyboard.is_pressed(*key),
                InputBinding::KeyWithModifiers(key, modifiers) => {
                    keyboard.is_pressed(*key) && keyboard.modifiers().contains(*modifiers)
                }
                InputBinding::Mouse(button) => mouse.is_pressed(*button),
            });

            // Check if any binding was just pressed
            let any_just_pressed = action.bindings.iter().any(|binding| match binding {
                InputBinding::Key(key) => keyboard.is_just_pressed(*key),
                InputBinding::KeyWithModifiers(key, modifiers) => {
                    keyboard.is_just_pressed(*key) && keyboard.modifiers().contains(*modifiers)
                }
                InputBinding::Mouse(button) => mouse.is_just_pressed(*button),
            });

            // Check if any binding was just released (and none are still pressed)
            let any_just_released = !any_pressed
                && action.bindings.iter().any(|binding| match binding {
                    InputBinding::Key(key) | InputBinding::KeyWithModifiers(key, _) => {
                        keyboard.is_just_released(*key)
                    }
                    InputBinding::Mouse(button) => mouse.is_just_released(*button),
                });

            // Update action state
            if any_just_pressed && !action.state.is_pressed() {
                action.state = ButtonState::JustPressed;
            } else if any_just_released {
                action.state = ButtonState::JustReleased;
            } else if any_pressed && action.state == ButtonState::JustPressed {
                action.state = ButtonState::Pressed;
            } else if !any_pressed && action.state == ButtonState::JustReleased {
                action.state = ButtonState::Released;
            }
        }
    }

    /// Returns `true` if the action is currently pressed.
    #[must_use]
    pub fn is_pressed(&self, action: &str) -> bool {
        self.actions
            .get(action)
            .is_some_and(|a| a.state.is_pressed())
    }

    /// Returns `true` if the action was just pressed this frame.
    #[must_use]
    pub fn is_just_pressed(&self, action: &str) -> bool {
        self.actions
            .get(action)
            .is_some_and(|a| a.state.is_just_pressed())
    }

    /// Returns `true` if the action was just released this frame.
    #[must_use]
    pub fn is_just_released(&self, action: &str) -> bool {
        self.actions
            .get(action)
            .is_some_and(|a| a.state.is_just_released())
    }

    /// Called at end of frame to transition action states.
    pub fn end_frame(&mut self) {
        for action in self.actions.values_mut() {
            action.state.end_frame();
        }
    }
}

/// Builder for creating an action map with a fluent API.
#[derive(Debug, Default)]
pub struct ActionMapBuilder {
    actions: ActionMap,
}

impl ActionMapBuilder {
    /// Create a new builder.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a binding to an action.
    #[must_use]
    pub fn bind(mut self, action: impl Into<String>, binding: impl Into<InputBinding>) -> Self {
        self.actions.bind(action, binding);
        self
    }

    /// Add multiple bindings to an action.
    #[must_use]
    pub fn bind_many(
        mut self,
        action: impl Into<String>,
        bindings: impl IntoIterator<Item = impl Into<InputBinding>>,
    ) -> Self {
        let action_name = action.into();
        for binding in bindings {
            self.actions.bind(action_name.clone(), binding);
        }
        self
    }

    /// Build the action map.
    #[must_use]
    pub fn build(self) -> ActionMap {
        self.actions
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn action_binding() {
        let mut actions = ActionMap::new();
        actions.bind("jump", KeyCode::Space);
        actions.bind("jump", KeyCode::KeyW);

        let bindings = actions.get_bindings("jump").unwrap();
        assert_eq!(bindings.len(), 2);
    }

    #[test]
    fn action_builder() {
        let actions = ActionMap::builder()
            .bind("move_forward", KeyCode::KeyW)
            .bind("move_forward", KeyCode::ArrowUp)
            .bind("jump", KeyCode::Space)
            .build();

        assert_eq!(actions.get_bindings("move_forward").unwrap().len(), 2);
        assert_eq!(actions.get_bindings("jump").unwrap().len(), 1);
    }

    #[test]
    fn unbind() {
        let mut actions = ActionMap::new();
        actions.bind("test", KeyCode::KeyA);
        actions.bind("test", KeyCode::KeyB);

        actions.unbind("test", InputBinding::Key(KeyCode::KeyA));
        let bindings = actions.get_bindings("test").unwrap();
        assert_eq!(bindings.len(), 1);
        assert_eq!(bindings[0], InputBinding::Key(KeyCode::KeyB));
    }
}
