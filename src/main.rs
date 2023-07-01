use vulkano_win::VkSurfaceBuild;
use winit::event_loop::EventLoop;
use winit::window::WindowBuilder;

pub mod camera;
pub mod model;
pub mod vulkano_wrapper;
pub mod window;

fn main() {
    let instance = vulkano_wrapper::get_instance();
    let event_loop = EventLoop::new();
    let surface = WindowBuilder::new()
        .with_title("Vulkano 3D Model Viewer")
        .build_vk_surface(&event_loop, instance.clone())
        .unwrap();
    let (physical_device, device, queue) = vulkano_wrapper::instantiate_device(&instance, &surface);
    let memory_allocator = vulkano_wrapper::memory_allocator::new(device.clone());

    let model = model::Model::new("models/viking_room.obj", "textures/viking_room.png");
    let mut image_builder =
        vulkano_wrapper::command_buffer::create_command_buffer_builder(&memory_allocator, &queue);
    let vulkano_model =
        vulkano_wrapper::model::VulkanoModel::new(model, &memory_allocator, &mut image_builder);

    vulkano_wrapper::run_event_loop(
        event_loop,
        physical_device,
        surface,
        device,
        queue,
        vulkano_model,
        image_builder,
        memory_allocator,
    );
}
