use cgmath::{Point2, Point3, Vector3};
use std::env;
use vulkano_win::VkSurfaceBuild;
use winit::event_loop::EventLoop;
use winit::window::WindowBuilder;

pub mod camera;
pub mod config;
pub mod model;
pub mod vulkano_wrapper;
pub mod window;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        panic!("You need to pass a .json file available in the configs folder!");
    }
    let config_filename = &args[1];
    let config = config::read_file(config_filename);
    let model = model::Model::new(&config.scene.model_filename, &config.scene.texture_filename);
    let mut camera = crate::camera::Camera::new(
        Point3::new(
            config.scene.camera.position.x,
            config.scene.camera.position.y,
            config.scene.camera.position.z,
        ),
        Point3::new(
            config.scene.camera.target.x,
            config.scene.camera.target.y,
            config.scene.camera.target.z,
        ),
        Vector3::new(
            config.scene.camera.up.x,
            config.scene.camera.up.y,
            config.scene.camera.up.z,
        ),
    );
    camera.set_window_size(Point2::new(
        config.window.width as u32,
        config.window.height as u32,
    ));

    let instance = vulkano_wrapper::get_instance();
    let event_loop = EventLoop::new();
    let surface = WindowBuilder::new()
        .with_title("Vulkano 3D Model Viewer")
        .build_vk_surface(&event_loop, instance.clone())
        .unwrap();
    let (physical_device, device, queue) = vulkano_wrapper::instantiate_device(&instance, &surface);
    let memory_allocator = vulkano_wrapper::memory_allocator::new(device.clone());

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
        camera,
    );
}
