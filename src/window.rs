use cgmath::Point2;

use winit::dpi::PhysicalPosition;
use winit::event::{
    ElementState, KeyboardInput, ModifiersState, MouseButton, MouseScrollDelta, VirtualKeyCode,
    WindowEvent,
};
use winit::event_loop::ControlFlow;

use crate::camera::{Camera, CameraMouseButton, CameraMovement, MouseModifierFlags};

fn exit_window(control_flow: &mut ControlFlow) {
    *control_flow = ControlFlow::Exit;
}

fn resize_window(has_window_resized: &mut bool) {
    *has_window_resized = true;
}

fn process_keyboard(input: KeyboardInput, camera: &mut Camera, control_flow: &mut ControlFlow) {
    if input.state == ElementState::Pressed {
        match input.virtual_keycode {
            Some(VirtualKeyCode::W) => camera.process_keyboard(CameraMovement::Forward),
            Some(VirtualKeyCode::S) => camera.process_keyboard(CameraMovement::Backward),
            Some(VirtualKeyCode::A) => camera.process_keyboard(CameraMovement::Left),
            Some(VirtualKeyCode::D) => camera.process_keyboard(CameraMovement::Right),
            Some(VirtualKeyCode::C) => camera.print_camera_data(),
            Some(VirtualKeyCode::Escape) => exit_window(control_flow),
            _ => (),
        }
    }
}

fn process_mouse_input(
    state: ElementState,
    button: MouseButton,
    camera: &mut Camera,
    mouse_button: &mut CameraMouseButton,
    mouse_position: &Point2<i32>,
) {
    *mouse_button = CameraMouseButton::None;
    if state == ElementState::Pressed {
        match button {
            MouseButton::Left => *mouse_button = CameraMouseButton::Left,
            MouseButton::Middle => *mouse_button = CameraMouseButton::Middle,
            MouseButton::Right => *mouse_button = CameraMouseButton::Right,
            _ => (),
        }
    }
    camera.set_mouse_position(*mouse_position);
}

fn process_mouse_modifiers(modifiers: ModifiersState, mouse_modifiers: &mut MouseModifierFlags) {
    match modifiers {
        ModifiersState::ALT => *mouse_modifiers = MouseModifierFlags::Alt,
        ModifiersState::CTRL => *mouse_modifiers = MouseModifierFlags::Ctrl,
        ModifiersState::SHIFT => *mouse_modifiers = MouseModifierFlags::Shift,
        _ => *mouse_modifiers = MouseModifierFlags::None,
    }
}

fn process_cursor_movement(
    position: PhysicalPosition<f64>,
    camera: &mut Camera,
    mouse_button: &CameraMouseButton,
    mouse_position: &mut Point2<i32>,
    mouse_modifiers: &mut MouseModifierFlags,
) {
    mouse_position.x = position.x as i32;
    mouse_position.y = position.y as i32;
    match mouse_button {
        CameraMouseButton::None => (),
        _ => camera.process_mouse_movement(*mouse_position, *mouse_button, *mouse_modifiers),
    }
}

fn process_mouse_wheel(delta: MouseScrollDelta, camera: &mut Camera) {
    match delta {
        MouseScrollDelta::LineDelta(_x, y) => {
            camera.process_mouse_scroll(y as i32);
        }
        _ => (),
    }
}

pub fn process_event(
    event: WindowEvent,
    control_flow: &mut ControlFlow,
    has_window_resized: &mut bool,
    camera: &mut Camera,
    mouse_button: &mut CameraMouseButton,
    mouse_position: &mut Point2<i32>,
    mouse_modifiers: &mut MouseModifierFlags,
) {
    match event {
        WindowEvent::CloseRequested => exit_window(control_flow),
        WindowEvent::Resized(_) => resize_window(has_window_resized),
        WindowEvent::KeyboardInput {
            device_id: _,
            input,
            ..
        } => process_keyboard(input, camera, control_flow),
        WindowEvent::MouseInput {
            device_id: _,
            state,
            button,
            ..
        } => process_mouse_input(state, button, camera, mouse_button, mouse_position),
        WindowEvent::ModifiersChanged(modifiers) => {
            process_mouse_modifiers(modifiers, mouse_modifiers)
        }
        WindowEvent::CursorMoved {
            device_id: _,
            position,
            ..
        } => process_cursor_movement(
            position,
            camera,
            mouse_button,
            mouse_position,
            mouse_modifiers,
        ),
        WindowEvent::MouseWheel {
            device_id: _,
            delta,
            ..
        } => process_mouse_wheel(delta, camera),
        _ => (),
    }
}
