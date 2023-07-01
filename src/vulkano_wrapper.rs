use cgmath::{Point2, Point3, Vector3};

use std::sync::Arc;

use vulkano::buffer::BufferContents;
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, PrimaryAutoCommandBuffer, PrimaryCommandBufferAbstract,
};
use vulkano::device::physical::{PhysicalDevice, PhysicalDeviceType};
use vulkano::device::{
    Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo, QueueFlags,
};
use vulkano::format::Format;
use vulkano::image::view::ImageView;
use vulkano::image::{AttachmentImage, ImageAccess, ImageUsage, SwapchainImage};
use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::pipeline::graphics::depth_stencil::DepthStencilState;
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::vertex_input::Vertex;
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::GraphicsPipeline;
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass};
use vulkano::shader::ShaderModule;
use vulkano::swapchain::{
    self, AcquireError, Surface, Swapchain, SwapchainCreateInfo, SwapchainCreationError,
    SwapchainPresentInfo,
};
use vulkano::sync::{self, FlushError, GpuFuture};
use vulkano::VulkanLibrary;

use winit::event::Event;
use winit::event_loop::EventLoop;
use winit::window::Window;

use crate::camera::Camera;
use crate::window::MouseHandler;
use memory_allocator::MemoryAllocator;

use self::model::VulkanoModel;

pub mod command_buffer;
pub mod memory_allocator;
pub mod model;
pub mod shader;

#[derive(BufferContents, Vertex)]
#[repr(C)]
pub struct CustomVertex {
    #[format(R32G32B32_SFLOAT)]
    pub position: [f32; 3],
    #[format(R32G32B32_SFLOAT)]
    pub normal: [f32; 3],
    #[format(R32G32B32_SFLOAT)]
    pub texture_coords: [f32; 3],
}

fn get_device_extensions() -> DeviceExtensions {
    DeviceExtensions {
        khr_swapchain: true,
        ..DeviceExtensions::empty()
    }
}

pub fn get_instance() -> Arc<Instance> {
    let library = VulkanLibrary::new().expect("No local Vulkan library/DLL");
    let required_extensions = vulkano_win::required_extensions(&library);
    Instance::new(
        library,
        InstanceCreateInfo {
            enabled_extensions: required_extensions,
            enumerate_portability: true,
            ..Default::default()
        },
    )
    .expect("Failed to create Instance")
}

fn select_physical_device(
    instance: &Arc<Instance>,
    surface: &Arc<Surface>,
) -> (Arc<PhysicalDevice>, u32) {
    instance
        .enumerate_physical_devices()
        .expect("Could not enumerate physical devices")
        .filter(|p| p.supported_extensions().contains(&get_device_extensions()))
        .filter_map(|p| {
            p.queue_family_properties()
                .iter()
                .enumerate()
                .position(|(i, q)| {
                    q.queue_flags.contains(QueueFlags::GRAPHICS)
                        && p.surface_support(i as u32, &surface).unwrap_or(false)
                })
                .map(|q| (p, q as u32))
        })
        .min_by_key(|(p, _)| match p.properties().device_type {
            PhysicalDeviceType::DiscreteGpu => 0,
            PhysicalDeviceType::IntegratedGpu => 1,
            PhysicalDeviceType::VirtualGpu => 2,
            PhysicalDeviceType::Cpu => 3,
            _ => 4,
        })
        .expect("No devices available")
}

pub fn instantiate_device(
    instance: &Arc<Instance>,
    surface: &Arc<Surface>,
) -> (Arc<PhysicalDevice>, Arc<Device>, Arc<Queue>) {
    let (physical_device, queue_family_index) = select_physical_device(&instance, &surface);
    let (device, mut queues) = Device::new(
        physical_device.clone(),
        DeviceCreateInfo {
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
            enabled_extensions: get_device_extensions(),
            ..Default::default()
        },
    )
    .expect("Failed to create device");
    (physical_device, device, queues.next().unwrap())
}

fn get_image_format(
    physical_device: &Arc<PhysicalDevice>,
    surface: &Arc<Surface>,
) -> Option<Format> {
    Some(
        physical_device
            .surface_formats(&surface, Default::default())
            .unwrap()[0]
            .0,
    )
}

fn create_swapchain(
    device: &Arc<Device>,
    physical_device: &Arc<PhysicalDevice>,
    surface: &Arc<Surface>,
) -> (Arc<Swapchain>, Vec<Arc<SwapchainImage>>) {
    let capabilities = physical_device
        .surface_capabilities(&surface, Default::default())
        .expect("Failed to get surface capabilities");
    Swapchain::new(
        device.clone(),
        surface.clone(),
        SwapchainCreateInfo {
            min_image_count: capabilities.min_image_count + 1,
            image_format: get_image_format(&physical_device, &surface),
            image_extent: get_window(&surface).inner_size().into(),
            image_usage: ImageUsage::COLOR_ATTACHMENT,
            composite_alpha: capabilities
                .supported_composite_alpha
                .into_iter()
                .next()
                .unwrap(),
            ..Default::default()
        },
    )
    .unwrap()
}

fn get_render_pass(device: Arc<Device>, swapchain: &Arc<Swapchain>) -> Arc<RenderPass> {
    vulkano::single_pass_renderpass!(
        device,
        attachments: {
            color: {
                load: Clear,
                store: Store,
                format: swapchain.image_format(),
                samples: 1
            },
            depth_stencil: {
                load: Clear,
                store: DontCare,
                format: Format::D16_UNORM,
                samples: 1,
            },
        },
        pass: {
            color: [color],
            depth_stencil: {depth_stencil},
        },
    )
    .unwrap()
}

fn get_color_buffer(image: Arc<SwapchainImage>) -> Arc<ImageView<SwapchainImage>> {
    ImageView::new_default(image).unwrap()
}

fn get_depth_buffer(
    dimensions: [u32; 2],
    memory_allocator: &MemoryAllocator,
) -> Arc<ImageView<AttachmentImage>> {
    ImageView::new_default(
        AttachmentImage::transient(&memory_allocator.standard, dimensions, Format::D16_UNORM)
            .unwrap(),
    )
    .unwrap()
}

fn get_framebuffers(
    images: &[Arc<SwapchainImage>],
    render_pass: &Arc<RenderPass>,
    memory_allocator: &MemoryAllocator,
) -> Vec<Arc<Framebuffer>> {
    images
        .iter()
        .map(|image| {
            Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![
                        get_color_buffer(image.clone()),
                        get_depth_buffer(images[0].dimensions().width_height(), memory_allocator)
                            .clone(),
                    ],
                    ..Default::default()
                },
            )
            .unwrap()
        })
        .collect::<Vec<_>>()
}

fn get_pipeline(
    device: Arc<Device>,
    vertex_shader: Arc<ShaderModule>,
    fragment_shader: Arc<ShaderModule>,
    render_pass: Arc<RenderPass>,
    viewport: Viewport,
) -> Arc<GraphicsPipeline> {
    GraphicsPipeline::start()
        .vertex_input_state([CustomVertex::per_vertex()])
        .vertex_shader(vertex_shader.entry_point("main").unwrap(), ())
        .input_assembly_state(InputAssemblyState::new())
        .viewport_state(ViewportState::viewport_fixed_scissor_irrelevant([viewport]))
        .depth_stencil_state(DepthStencilState::simple_depth_test())
        .fragment_shader(fragment_shader.entry_point("main").unwrap(), ())
        .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
        .build(device.clone())
        .unwrap()
}

fn get_window(surface: &Arc<Surface>) -> Arc<Window> {
    surface
        .object()
        .unwrap()
        .clone()
        .downcast::<Window>()
        .unwrap()
}

fn get_viewport(dimensions: Option<[f32; 2]>) -> Viewport {
    Viewport {
        origin: [0.0, 0.0],
        dimensions: match dimensions {
            None => [1024.0, 1024.0],
            Some(value) => value,
        },
        depth_range: 0.0..1.0,
    }
}

pub fn run_event_loop(
    event_loop: EventLoop<()>,
    physical_device: Arc<PhysicalDevice>,
    surface: Arc<Surface>,
    device: Arc<Device>,
    queue: Arc<Queue>,
    vulkano_model: VulkanoModel,
    image_builder: AutoCommandBufferBuilder<
        PrimaryAutoCommandBuffer,
        Arc<StandardCommandBufferAllocator>,
    >,
    memory_allocator: MemoryAllocator,
) {
    let mut camera = Camera::new(
        Point3::new(2.33, 1.40, 1.72),
        Point3::new(-0.06, -0.08, 0.17),
        Vector3::new(-0.42337, -0.228691, 0.876617),
    );
    let viewport = get_viewport(None);
    camera.set_window_size(Point2::new(
        viewport.dimensions[0] as i32,
        viewport.dimensions[1] as i32,
    ));
    let mut recreate_swapchain = false;
    let mut previous_frame_end = Some(
        image_builder
            .build()
            .unwrap()
            .execute(queue.clone())
            .unwrap()
            .boxed(),
    );
    let (vertex_shader, fragment_shader) = shader::load(device.clone());
    let (mut swapchain, images) = create_swapchain(&device, &physical_device, &surface);
    let render_pass = get_render_pass(device.clone(), &swapchain);
    let mut framebuffers = get_framebuffers(&images, &render_pass, &memory_allocator);

    let mut pipeline = get_pipeline(
        device.clone(),
        vertex_shader.clone(),
        fragment_shader.clone(),
        render_pass.clone(),
        get_viewport(None),
    );

    let mut mouse_handler = MouseHandler::new();
    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent { event, .. } => crate::window::process_event(
            event,
            control_flow,
            &mut recreate_swapchain,
            &mut camera,
            &mut mouse_handler,
        ),
        Event::RedrawEventsCleared => {
            previous_frame_end.as_mut().unwrap().cleanup_finished();
            if recreate_swapchain {
                recreate_swapchain = false;
                let (new_swapchain, new_images) = match swapchain.recreate(SwapchainCreateInfo {
                    image_extent: get_window(&surface).inner_size().into(),
                    ..swapchain.create_info()
                }) {
                    Ok(r) => r,
                    Err(SwapchainCreationError::ImageExtentNotSupported { .. }) => return,
                    Err(e) => panic!("Failed to recreate swapchain: {e}"),
                };
                swapchain = new_swapchain;
                framebuffers = get_framebuffers(&new_images, &render_pass, &memory_allocator);

                pipeline = get_pipeline(
                    device.clone(),
                    vertex_shader.clone(),
                    fragment_shader.clone(),
                    render_pass.clone(),
                    get_viewport(Some(get_window(&surface).inner_size().into())),
                );
            }

            let (image_i, suboptimal, acquire_future) =
                match swapchain::acquire_next_image(swapchain.clone(), None) {
                    Ok(r) => r,
                    Err(AcquireError::OutOfDate) => {
                        recreate_swapchain = true;
                        return;
                    }
                    Err(e) => panic!("Failed to acquire next image: {e}"),
                };

            if suboptimal {
                recreate_swapchain = true;
            }

            camera.set_window_size(Point2::new(
                swapchain.image_extent()[0] as i32,
                swapchain.image_extent()[1] as i32,
            ));

            let command_buffer = command_buffer::get_command_buffer(
                &queue,
                &pipeline,
                &framebuffers[image_i as usize],
                &vulkano_model,
                &device.clone(),
                &memory_allocator,
                &camera,
            );

            let future = previous_frame_end
                .take()
                .unwrap()
                .join(acquire_future)
                .then_execute(queue.clone(), command_buffer)
                .unwrap()
                .then_swapchain_present(
                    queue.clone(),
                    SwapchainPresentInfo::swapchain_image_index(swapchain.clone(), image_i),
                )
                .then_signal_fence_and_flush();

            match future {
                Ok(future) => {
                    previous_frame_end = Some(future.boxed());
                }
                Err(FlushError::OutOfDate) => {
                    recreate_swapchain = true;
                    previous_frame_end = Some(sync::now(device.clone()).boxed());
                }
                Err(e) => {
                    println!("failed to flush future: {e}");
                    previous_frame_end = Some(sync::now(device.clone()).boxed());
                }
            }
        }
        _ => (),
    });
}
