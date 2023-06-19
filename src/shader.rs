use std::sync::Arc;
use vulkano::device::Device;
use vulkano::shader::ShaderModule;

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: "
            #version 460

            layout(location = 0) in vec2 position;
            
            void main() {
                gl_Position = vec4(position, 0.0, 1.0);
            }
        "
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: "
            #version 460

            layout(location = 0) out vec4 f_color;
            
            void main() {
                f_color = vec4(1.0, 0.0, 0.0, 1.0);
            }
        "
    }
}

pub fn load(device: Arc<Device>) -> (Arc<ShaderModule>, Arc<ShaderModule>) {
    let vertex_shader = vs::load(device.clone()).expect("Failed to create shader module");
    let fragment_shader = fs::load(device.clone()).expect("Failed to create shader module");
    (vertex_shader, fragment_shader)
}
