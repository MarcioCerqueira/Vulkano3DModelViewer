use vulkano_win::VkSurfaceBuild;

use winit::event_loop::EventLoop;
use winit::window::WindowBuilder;

pub mod model;
pub mod shader;
pub mod vulkano_wrapper;

fn main() {
    let instance = vulkano_wrapper::get_instance();
    let event_loop = EventLoop::new();
    let surface = WindowBuilder::new()
        .build_vk_surface(&event_loop, instance.clone())
        .unwrap();
    let (physical_device, device, queue) = vulkano_wrapper::instantiate_device(&instance, &surface);
    let (swapchain, images) =
        vulkano_wrapper::create_swapchain(&device, &physical_device, &surface);
    let render_pass = vulkano_wrapper::get_render_pass(device.clone(), &swapchain);
    let framebuffers = vulkano_wrapper::get_framebuffers(device.clone(), &images, &render_pass);

    let (vertex_shader, fragment_shader) = shader::load(device.clone());
    let viewport = vulkano_wrapper::get_viewport();
    let pipeline = vulkano_wrapper::get_pipeline(
        device.clone(),
        vertex_shader.clone(),
        fragment_shader.clone(),
        render_pass.clone(),
        viewport.clone(),
    );

    let model = model::Model::new("models/viking_room.obj");
    let vertex_buffer = vulkano_wrapper::create_vertex_buffer(device.clone(), model.vertices);
    let index_buffer = vulkano_wrapper::create_index_buffer(device.clone(), model.indices);

    let command_buffers = vulkano_wrapper::get_command_buffers(
        device.clone(),
        &queue,
        &pipeline,
        &framebuffers,
        &vertex_buffer,
        &index_buffer,
    );

    vulkano_wrapper::run_event_loop(
        images.len(),
        event_loop,
        swapchain,
        surface,
        render_pass,
        viewport,
        device,
        vertex_shader,
        fragment_shader,
        command_buffers,
        queue,
        vertex_buffer,
        index_buffer,
    );
}
