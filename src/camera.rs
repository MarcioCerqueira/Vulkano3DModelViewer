use bitflags::bitflags;
use cgmath::{InnerSpace, Matrix4, Point2, Point3, Rad, Vector2, Vector3, Vector4, Zero};

#[derive(PartialEq)]
pub enum Mode {
    Examine,
    Fly,
    Walk,
    Trackball,
}

pub enum CameraMovement {
    Forward,
    Backward,
    Left,
    Right,
}

pub enum MouseButton {
    None,
    Left,
    Middle,
    Right,
}

pub enum Action {
    None,
    Orbit,
    Dolly,
    Pan,
    LookAround,
}

bitflags! {
    #[derive(Clone, Copy, PartialEq)]
    struct MouseModifierFlags: u32 {
        const None = 0b00000000;
        const Shift = 0b00000001;
        const Ctrl = 0b00000010;
        const Alt = 0b00000100;
    }
}

pub struct Camera {
    eye: Point3<f32>,
    center: Point3<f32>,
    up: Vector3<f32>,
    roll: f32, // Rotation around the Z axis in RAD
    matrix: Matrix4<f32>,
    window_size: Point2<i32>,
    speed: f32,
    movement_speed: f32,
    zoom: f32,
    mouse_position: Point2<i32>,
    mode: Mode,
    trackball_size: f32,
}

impl Camera {
    pub fn new(
        camera_position: Point3<f32>,
        center_position: Point3<f32>,
        up_vector: Vector3<f32>,
    ) -> Camera {
        let mut camera = Camera {
            eye: camera_position,
            center: center_position,
            up: up_vector,
            roll: 0.0,
            matrix: Matrix4::new(
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            ),
            window_size: Point2 { x: 1, y: 1 },
            speed: 30.0,
            movement_speed: 0.1,
            zoom: 45.0,
            mouse_position: Point2 { x: 0, y: 0 },
            mode: Mode::Examine,
            trackball_size: 0.8,
        };
        Self::update(&mut camera);
        camera
    }

    pub fn set_window_size(&mut self, window_size: Point2<i32>) {
        self.window_size = window_size;
    }

    pub fn get_eye(&self) -> Point3<f32> {
        self.eye
    }

    fn update(camera: &mut Camera) {
        camera.matrix = Matrix4::look_at_rh(camera.eye, camera.center, camera.up);
        if !camera.roll.is_zero() {
            camera.matrix = camera.matrix * Matrix4::from_angle_z(Rad(camera.roll))
        }
    }

    fn process_mouse_movement(
        &mut self,
        position: Point2<i32>,
        mouse_button: MouseButton,
        modifiers: MouseModifierFlags,
    ) -> Action {
        let current_action = match mouse_button {
            MouseButton::Left => {
                if ((modifiers & MouseModifierFlags::Ctrl != MouseModifierFlags::None)
                    && (modifiers & MouseModifierFlags::Shift != MouseModifierFlags::None))
                    || (modifiers & MouseModifierFlags::Alt != MouseModifierFlags::None)
                {
                    if self.mode == Mode::Examine {
                        Action::LookAround
                    } else {
                        Action::Orbit
                    }
                } else if modifiers & MouseModifierFlags::Shift != MouseModifierFlags::None {
                    Action::Dolly
                } else if modifiers & MouseModifierFlags::Ctrl != MouseModifierFlags::None {
                    Action::Pan
                } else {
                    if self.mode == Mode::Examine {
                        Action::Orbit
                    } else {
                        Action::LookAround
                    }
                }
            }
            MouseButton::Middle => Action::Pan,
            MouseButton::Right => Action::Dolly,
            _ => Action::None,
        };
        self.motion(position, &current_action);
        current_action
    }

    fn motion(&mut self, position: Point2<i32>, action: &Action) {
        let delta = position - self.mouse_position;
        let delta: Vector2<f32> = Vector2::new(
            delta.x as f32 / self.window_size.x as f32,
            delta.y as f32 / self.window_size.y as f32,
        );
        match action {
            Action::Orbit => {
                let invert = self.mode == Mode::Trackball;
                self.orbit(delta, invert);
            }
            Action::Dolly => {
                self.dolly(delta);
            }
            Action::Pan => {
                self.pan(delta);
            }
            Action::LookAround => match self.mode {
                Mode::Trackball => self.trackball(position),
                _ => self.orbit(delta, true),
            },
            _ => {}
        }
        Self::update(self);
        self.mouse_position = position;
    }

    fn set_roll(&mut self, roll: f32) {
        self.roll = roll;
        Self::update(self);
    }

    fn process_mouse_scroll(&mut self, value: i32) {
        let f_value = value as f32;
        let mut dx = f_value * f_value.abs() / self.window_size.x as f32;

        dx = dx * self.speed;
        self.dolly(Vector2::new(dx, dx));
        Self::update(self);
    }

    fn dolly(&mut self, delta: Vector2<f32>) {
        let mut z = self.center - self.eye;
        let mut length = z.dot(z).sqrt();

        // We are at the point of interest, and don't know any direction, so do nothing!
        if length.is_zero() {
            return;
        }

        // Use the larger movement.
        let dd = match self.mode {
            Mode::Examine => -delta.y,
            _ => match delta.x.abs() > delta.y.abs() {
                true => delta.x,
                false => -delta.y,
            },
        };

        let mut factor = self.speed * dd / length;

        // Adjust speed based on distance.
        length = length / 10.0;
        length = match length < 0.001 {
            true => 0.001,
            false => length,
        };
        factor = factor * length;

        // Don't move to or through the point of interest.
        if 1.0 <= factor {
            return;
        }

        z = z * factor;

        // Not going up
        if self.mode == Mode::Walk {
            if self.up.y > self.up.z {
                z.y = 0.0;
            } else {
                z.z = 0.0;
            }
        }

        self.eye = self.eye + z;

        // In fly mode, the interest moves with us.
        if self.mode != Mode::Examine {
            self.center = self.center + z;
        }
    }

    fn orbit(&mut self, delta: Vector2<f32>, invert: bool) {
        if delta.is_zero() {
            return;
        }

        // Full width will do a full turn
        let d = delta * std::f32::consts::PI * 2.0;

        // Get the camera
        let origin = match invert {
            true => self.eye,
            false => self.center,
        };
        let position = match invert {
            true => self.center,
            false => self.eye,
        };

        // Get the length of sight
        let mut center_to_eye = position - origin;
        let radius = center_to_eye.dot(center_to_eye).sqrt();
        center_to_eye = center_to_eye.normalize();

        // Find the rotation around the UP axis (Y)
        let z_axis = center_to_eye;
        let y_rotation = Matrix4::from_axis_angle(self.up, Rad(-d.x));

        // Apply the (Y) rotation to the eye-center vector
        let mut tmp_vector =
            y_rotation * Vector4::new(center_to_eye.x, center_to_eye.y, center_to_eye.z, 0.0);
        center_to_eye = Vector3::new(tmp_vector.x, tmp_vector.y, tmp_vector.z);

        // Find the rotation around the X vector: cross between eye-center and up (X)
        let mut x_axis = self.up.cross(z_axis);
        x_axis = x_axis.normalize();
        let x_rotation = Matrix4::from_axis_angle(x_axis, Rad(-d.y));

        // Apply the (X) rotation to the eye-center vector
        tmp_vector =
            x_rotation * Vector4::new(center_to_eye.x, center_to_eye.y, center_to_eye.z, 0.0);
        let rotated_vector = Vector3::new(tmp_vector.x, tmp_vector.y, tmp_vector.z);
        if rotated_vector.x.signum() == center_to_eye.x.signum() {
            center_to_eye = rotated_vector;
        }

        // Make the vector as long as it was originally
        center_to_eye = center_to_eye * radius;

        // Finding the new position
        let new_position = center_to_eye + Vector3::new(origin.x, origin.y, origin.z);
        let new_position = Point3::new(new_position.x, new_position.y, new_position.z);
        match invert {
            true => self.eye = new_position, // Normal: change the position of the camera
            false => self.center = new_position, // Inverted: change the interest point
        }
    }

    fn pan(&mut self, delta: Vector2<f32>) {
        let mut z = self.eye - self.center;
        let length = z.dot(z).sqrt() / 0.785; // 45 degrees
        z = z.normalize();
        let mut x = self.up.cross(z).normalize();
        let mut y = z.cross(x).normalize();
        x = x * -delta.x * length;
        y = y * delta.y * length;

        if self.mode == Mode::Fly {
            x = -x;
            y = -y;
        }

        self.eye = self.eye + x + y;
        self.center = self.center + x + y;
    }

    fn project_onto_tb_sphere(&self, p: Vector2<f32>) -> f32 {
        let d = p.dot(p).sqrt();
        if d < self.trackball_size * 0.70710678118654752440 {
            (self.trackball_size * self.trackball_size - d * d).sqrt()
        } else {
            let t = self.trackball_size / 1.41421356237309504880;
            t * t / d
        }
    }

    fn trackball(&mut self, position: Point2<i32>) {
        let p_zero = Vector2::new(
            2.0 * (self.mouse_position.x - self.window_size.x / 2) as f32
                / self.window_size.x as f32,
            2.0 * (self.window_size.y / 2 - self.mouse_position.y) as f32
                / self.window_size.y as f32,
        );
        let p_one = Vector2::new(
            2.0 * (position.x - self.window_size.x / 2) as f32 / self.window_size.x as f32,
            2.0 * (self.window_size.y / 2 - position.y) as f32 / self.window_size.y as f32,
        );
        // determine the z coordinate on the sphere
        let p_tb_zero = Vector3::new(p_zero.x, p_zero.y, self.project_onto_tb_sphere(p_zero));
        let p_tb_one = Vector3::new(p_one.x, p_one.y, self.project_onto_tb_sphere(p_one));
        // calculate the rotation axis via cross product between p0 and p1
        let axis = p_tb_zero.cross(p_tb_one).normalize();
        // calculate the angle
        let t = ((p_tb_zero - p_tb_one).dot(p_tb_zero - p_tb_one).sqrt()
            / (2.0 * self.trackball_size))
            .clamp(-1.0, 1.0);
        let rad = 2.0 * t.asin();
        let rot_axis = self.matrix * Vector4::new(axis.x, axis.y, axis.z, 0.0);
        let rot_mat =
            Matrix4::from_axis_angle(Vector3::new(rot_axis.x, rot_axis.y, rot_axis.z), Rad(rad));
        let pnt = self.eye - self.center;
        let pnt_two = rot_mat * Vector4::new(pnt.x, pnt.y, pnt.z, 1.0);
        let up_two = rot_mat * Vector4::new(self.up.x, self.up.y, self.up.z, 0.0);
        self.eye = Point3::new(
            self.center.x + pnt_two.x,
            self.center.y + pnt_two.y,
            self.center.z + pnt_two.z,
        );
        self.up = Vector3::new(up_two.x, up_two.y, up_two.z);
    }

    pub fn get_model_matrix(&self) -> Matrix4<f32> {
        Matrix4::from_angle_z(Rad(0.0))
    }

    pub fn get_view_matrix(&self) -> Matrix4<f32> {
        self.matrix
    }

    pub fn get_projection_matrix(&self) -> Matrix4<f32> {
        cgmath::perspective(
            Rad(self.zoom.to_radians()),
            self.window_size.x as f32 / self.window_size.y as f32,
            0.1,
            1000.0,
        )
    }

    pub fn print_camera_data(&self) {
        println!("=========== CAMERA DATA ===========");
        println!("Camera Position: {:?}", self.eye);
        println!("Target Position: {:?}", self.center);
        println!("Up Vector: {:?}", self.up);
    }

    pub fn process_keyboard(&mut self, direction: CameraMovement) {
        let front = (self.center - self.eye).normalize();
        let right = front.cross(self.up).normalize();
        self.up = right.cross(front).normalize();
        self.eye = match direction {
            CameraMovement::Forward => self.eye + front * self.movement_speed,
            CameraMovement::Backward => self.eye - front * self.movement_speed,
            CameraMovement::Left => self.eye - right * self.movement_speed,
            CameraMovement::Right => self.eye + right * self.movement_speed,
        };
        Self::update(self);
    }
}
