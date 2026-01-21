//! Input handling for the Voxelicous engine.
//!
//! This crate provides keyboard and mouse input handling with an action mapping system.
//! It is designed to be extensible for future input devices (gamepads, touch).
//!
//! # Core Types
//!
//! - [`InputManager`]: Main entry point combining keyboard, mouse, and actions
//! - [`KeyboardState`]: Tracks keyboard key states
//! - [`MouseState`]: Tracks mouse position, buttons, and scroll
//! - [`ActionMap`]: Maps named actions to input bindings
//!
//! # Usage
//!
//! ```ignore
//! use voxelicous_input::{InputManager, ActionMap};
//! use winit::keyboard::KeyCode;
//!
//! // Create input manager with actions
//! let actions = ActionMap::builder()
//!     .bind("move_forward", KeyCode::KeyW)
//!     .bind("move_forward", KeyCode::ArrowUp)
//!     .bind("move_back", KeyCode::KeyS)
//!     .bind("move_back", KeyCode::ArrowDown)
//!     .bind("move_left", KeyCode::KeyA)
//!     .bind("move_right", KeyCode::KeyD)
//!     .bind("jump", KeyCode::Space)
//!     .build();
//!
//! let mut input = InputManager::with_actions(actions);
//!
//! // In event handler
//! fn on_event(input: &mut InputManager, event: &WindowEvent) -> bool {
//!     input.process_window_event(event)
//! }
//!
//! // In update loop
//! fn update(input: &mut InputManager, dt: f32) {
//!     if input.is_action_pressed("move_forward") {
//!         // Move camera forward
//!     }
//!
//!     // Get mouse movement for camera rotation
//!     let mouse_delta = input.mouse_raw_delta();
//!
//!     // MUST call at end of update
//!     input.end_frame();
//! }
//! ```
//!
//! # Cursor Modes
//!
//! For FPS-style controls, use [`CursorMode::Locked`] to capture the cursor:
//!
//! ```ignore
//! // Toggle cursor lock on Escape
//! if input.is_key_just_pressed(KeyCode::Escape) {
//!     let new_mode = match input.cursor_mode() {
//!         CursorMode::Normal => CursorMode::Locked,
//!         CursorMode::Locked | CursorMode::Confined => CursorMode::Normal,
//!     };
//!     input.set_cursor_mode(new_mode);
//!
//!     // Apply to window
//!     window.set_cursor_grab(match new_mode {
//!         CursorMode::Normal => CursorGrabMode::None,
//!         CursorMode::Confined => CursorGrabMode::Confined,
//!         CursorMode::Locked => CursorGrabMode::Locked,
//!     });
//!     window.set_cursor_visible(new_mode == CursorMode::Normal);
//! }
//! ```

mod action;
mod button_state;
mod input;
mod keyboard;
mod modifiers;
mod mouse;

pub use action::{ActionMap, ActionMapBuilder, InputBinding};
pub use button_state::ButtonState;
pub use input::InputManager;
pub use keyboard::KeyboardState;
pub use modifiers::Modifiers;
pub use mouse::{CursorMode, MouseButton, MouseState};

// Re-export winit types commonly used with input
pub use winit::event::{DeviceEvent, WindowEvent};
pub use winit::keyboard::KeyCode;
