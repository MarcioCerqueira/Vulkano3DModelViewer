use cgmath::{Matrix4, Point3, Rad, Vector3};

#[derive(Default)]
pub struct Camera {}

impl Camera {
    pub fn get_projection_matrix(&self, aspect_ratio: f32) -> Matrix4<f32> {
        cgmath::perspective(Rad(std::f32::consts::FRAC_PI_2), aspect_ratio, 0.1, 1000.0)
    }

    pub fn get_view_matrix(&self) -> Matrix4<f32> {
        Matrix4::look_at_rh(
            Point3::new(0.75, 0.75, 2.0),
            Point3::new(0.25, 0.0, 1.0),
            Vector3::new(1.0, 1.0, 1.0),
        )
    }

    pub fn get_model_matrix(&self) -> Matrix4<f32> {
        Matrix4::from_angle_y(Rad(-std::f32::consts::FRAC_PI_6 as f32))
    }
}
