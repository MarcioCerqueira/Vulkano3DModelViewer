use std::sync::Arc;
use vulkano::device::Device;
use vulkano::shader::ShaderModule;

pub mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "shaders/vert.glsl",
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "shaders/frag.glsl",
    }
}

pub fn load(device: Arc<Device>) -> (Arc<ShaderModule>, Arc<ShaderModule>) {
    let vertex_shader = vs::load(device.clone()).expect("Failed to create shader module");
    let fragment_shader = fs::load(device.clone()).expect("Failed to create shader module");
    (vertex_shader, fragment_shader)
}
