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
    let memory_allocator = vulkano_wrapper::memory_allocator::new(device.clone());
    let framebuffers = vulkano_wrapper::get_framebuffers(&images, &render_pass, &memory_allocator);

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
    let vertex_buffer = vulkano_wrapper::create_vertex_buffer(model.vertices, &memory_allocator);
    let index_buffer = vulkano_wrapper::create_index_buffer(model.indices, &memory_allocator);

    vulkano_wrapper::run_event_loop(
        event_loop,
        swapchain,
        surface,
        render_pass,
        viewport,
        device,
        vertex_shader,
        fragment_shader,
        queue,
        vertex_buffer,
        index_buffer,
        pipeline,
        framebuffers,
        memory_allocator,
    );
}
