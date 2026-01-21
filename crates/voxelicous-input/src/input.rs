//! Main input manager combining keyboard, mouse, and action mapping.

use glam::Vec2;
use winit::event::{DeviceEvent, WindowEvent};
use winit::keyboard::KeyCode;

use crate::action::{ActionMap, InputBinding};
use crate::keyboard::KeyboardState;
use crate::modifiers::Modifiers;
use crate::mouse::{CursorMode, MouseButton, MouseState};

/// Input manager combining keyboard, mouse, and action mapping.
///
/// # Usage
///
/// ```ignore
/// // In your app struct
/// struct MyApp {
///     input: InputManager,
/// }
///
/// // In on_event
/// fn on_event(&mut self, event: &WindowEvent) -> bool {
///     self.input.process_window_event(event)
/// }
///
/// // In update
/// fn update(&mut self, ctx: &AppContext, dt: f32) {
///     if self.input.is_action_pressed("move_forward") {
///         // Move camera forward
///     }
///
///     // MUST call at end of update
///     self.input.end_frame();
/// }
/// ```
#[derive(Debug, Default)]
pub struct InputManager {
    /// Keyboard input state.
    keyboard: KeyboardState,
    /// Mouse input state.
    mouse: MouseState,
    /// Action mappings.
    actions: ActionMap,
}

impl InputManager {
    /// Create a new input manager.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a new input manager with pre-configured actions.
    #[must_use]
    pub fn with_actions(actions: ActionMap) -> Self {
        Self {
            keyboard: KeyboardState::new(),
            mouse: MouseState::new(),
            actions,
        }
    }

    /// Get a reference to the keyboard state.
    #[must_use]
    pub const fn keyboard(&self) -> &KeyboardState {
        &self.keyboard
    }

    /// Get a reference to the mouse state.
    #[must_use]
    pub const fn mouse(&self) -> &MouseState {
        &self.mouse
    }

    /// Get a mutable reference to the action map for runtime rebinding.
    pub fn actions_mut(&mut self) -> &mut ActionMap {
        &mut self.actions
    }

    /// Process a window event.
    ///
    /// Returns `true` if the event was consumed.
    pub fn process_window_event(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::KeyboardInput { event, .. } => {
                self.keyboard.process_key_event(event);
                true
            }
            WindowEvent::ModifiersChanged(modifiers) => {
                self.keyboard.set_modifiers(Modifiers::from(modifiers.state()));
                true
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.mouse.set_position(position.x, position.y);
                true
            }
            WindowEvent::MouseInput { button, state, .. } => {
                self.mouse.process_button(*button, *state);
                true
            }
            WindowEvent::MouseWheel { delta, .. } => {
                self.mouse.process_scroll(*delta);
                true
            }
            _ => false,
        }
    }

    /// Process a device event (for raw mouse motion).
    ///
    /// Call this from a `DeviceEvent` handler for accurate mouse motion in locked cursor mode.
    pub fn process_device_event(&mut self, event: &DeviceEvent) {
        if let DeviceEvent::MouseMotion { delta } = event {
            self.mouse.add_raw_motion(delta.0, delta.1);
        }
    }

    /// Update action states based on current keyboard/mouse state.
    ///
    /// **IMPORTANT**: Call this at the START of your update loop, before querying actions.
    pub fn update(&mut self) {
        self.actions.update(&self.keyboard, &self.mouse);
    }

    /// Called at the end of each frame to transition button states and reset deltas.
    ///
    /// **IMPORTANT**: This must be called at the end of every update cycle.
    pub fn end_frame(&mut self) {
        // Transition button states (JustPressed -> Pressed, JustReleased -> Released)
        self.keyboard.end_frame();
        self.mouse.end_frame();
        self.actions.end_frame();
    }

    // ===== Keyboard shortcuts =====

    /// Returns `true` if the key is currently pressed.
    #[must_use]
    pub fn is_key_pressed(&self, key: KeyCode) -> bool {
        self.keyboard.is_pressed(key)
    }

    /// Returns `true` if the key was just pressed this frame.
    #[must_use]
    pub fn is_key_just_pressed(&self, key: KeyCode) -> bool {
        self.keyboard.is_just_pressed(key)
    }

    /// Returns `true` if the key was just released this frame.
    #[must_use]
    pub fn is_key_just_released(&self, key: KeyCode) -> bool {
        self.keyboard.is_just_released(key)
    }

    /// Get the current modifier key state.
    #[must_use]
    pub const fn modifiers(&self) -> Modifiers {
        self.keyboard.modifiers()
    }

    // ===== Mouse shortcuts =====

    /// Get the current cursor position in window coordinates.
    #[must_use]
    pub const fn mouse_position(&self) -> Vec2 {
        self.mouse.position()
    }

    /// Get mouse movement delta since last frame.
    #[must_use]
    pub const fn mouse_delta(&self) -> Vec2 {
        self.mouse.delta()
    }

    /// Get raw mouse motion delta (useful for locked cursor mode).
    #[must_use]
    pub const fn mouse_raw_delta(&self) -> Vec2 {
        self.mouse.raw_delta()
    }

    /// Get scroll wheel delta.
    #[must_use]
    pub const fn scroll_delta(&self) -> Vec2 {
        self.mouse.scroll_delta()
    }

    /// Returns `true` if the mouse button is currently pressed.
    #[must_use]
    pub const fn is_mouse_pressed(&self, button: MouseButton) -> bool {
        self.mouse.is_pressed(button)
    }

    /// Returns `true` if the mouse button was just pressed this frame.
    #[must_use]
    pub const fn is_mouse_just_pressed(&self, button: MouseButton) -> bool {
        self.mouse.is_just_pressed(button)
    }

    /// Returns `true` if the mouse button was just released this frame.
    #[must_use]
    pub const fn is_mouse_just_released(&self, button: MouseButton) -> bool {
        self.mouse.is_just_released(button)
    }

    /// Get the current cursor mode.
    #[must_use]
    pub const fn cursor_mode(&self) -> CursorMode {
        self.mouse.cursor_mode()
    }

    /// Set the cursor mode.
    ///
    /// **Note**: This only updates the internal state. The caller is responsible
    /// for applying the mode to the actual window.
    pub fn set_cursor_mode(&mut self, mode: CursorMode) {
        self.mouse.set_cursor_mode(mode);
    }

    // ===== Action shortcuts =====

    /// Bind an input to an action.
    pub fn bind_action(&mut self, action: impl Into<String>, binding: impl Into<InputBinding>) {
        self.actions.bind(action, binding);
    }

    /// Returns `true` if the action is currently pressed.
    #[must_use]
    pub fn is_action_pressed(&self, action: &str) -> bool {
        self.actions.is_pressed(action)
    }

    /// Returns `true` if the action was just pressed this frame.
    #[must_use]
    pub fn is_action_just_pressed(&self, action: &str) -> bool {
        self.actions.is_just_pressed(action)
    }

    /// Returns `true` if the action was just released this frame.
    #[must_use]
    pub fn is_action_just_released(&self, action: &str) -> bool {
        self.actions.is_just_released(action)
    }

    /// Clear all input state.
    pub fn clear(&mut self) {
        self.keyboard.clear();
        self.mouse.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn input_manager_creation() {
        let input = InputManager::new();
        assert!(!input.is_key_pressed(KeyCode::KeyW));
        assert_eq!(input.mouse_position(), Vec2::ZERO);
    }

    #[test]
    fn input_manager_with_actions() {
        let actions = ActionMap::builder()
            .bind("test", KeyCode::Space)
            .build();

        let input = InputManager::with_actions(actions);
        assert!(!input.is_action_pressed("test"));
    }
}
