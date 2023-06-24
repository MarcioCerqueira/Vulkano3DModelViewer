use std::sync::Arc;
use vulkano::device::Device;
use vulkano::shader::ShaderModule;

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: "
            #version 460

            layout(location = 0) in vec3 position;
            layout(location = 1) in vec3 normal;
            
            layout(location = 0) out vec3 fragNormal;
            void main() {
                gl_Position = vec4(position, 1.0);
                fragNormal = normal;
            }
        "
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: "
            #version 460

            layout(location = 0) in vec3 fragNormal;
            layout(location = 0) out vec4 outColor;
            
            void main() {
                outColor = vec4(fragNormal, 1.0);
            }
        "
    }
}

pub fn load(device: Arc<Device>) -> (Arc<ShaderModule>, Arc<ShaderModule>) {
    let vertex_shader = vs::load(device.clone()).expect("Failed to create shader module");
    let fragment_shader = fs::load(device.clone()).expect("Failed to create shader module");
    (vertex_shader, fragment_shader)
}
