use vulkano_win::VkSurfaceBuild;
use winit::event_loop::EventLoop;
use winit::window::WindowBuilder;

pub mod camera;
pub mod model;
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

    let model = model::Model::new("models/viking_room.obj", "textures/viking_room.png");
    let mut image_builder =
        vulkano_wrapper::create_command_buffer_builder(&memory_allocator, &queue);
    let vulkano_model =
        vulkano_wrapper::model::VulkanoModel::new(model, &memory_allocator, &mut image_builder);

    vulkano_wrapper::run_event_loop(
        event_loop,
        swapchain,
        surface,
        render_pass,
        device,
        queue,
        vulkano_model,
        image_builder,
        framebuffers,
        memory_allocator,
    );
}
