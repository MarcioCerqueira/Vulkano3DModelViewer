use cgmath::{Matrix4, Point3, Rad, Vector3};

use std::sync::Arc;

use vulkano::buffer::allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo};
use vulkano::buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer, RenderPassBeginInfo,
    SubpassContents,
};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::device::physical::{PhysicalDevice, PhysicalDeviceType};
use vulkano::device::{
    Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo, QueueFlags,
};
use vulkano::format::Format;
use vulkano::image::view::ImageView;
use vulkano::image::{AttachmentImage, ImageAccess, ImageUsage, SwapchainImage};
use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryUsage, StandardMemoryAllocator};
use vulkano::pipeline::graphics::depth_stencil::DepthStencilState;
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::vertex_input::Vertex;
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::{GraphicsPipeline, Pipeline, PipelineBindPoint};
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass};
use vulkano::shader::ShaderModule;
use vulkano::swapchain::{
    self, AcquireError, Surface, Swapchain, SwapchainCreateInfo, SwapchainCreationError,
    SwapchainPresentInfo,
};
use vulkano::sync::{self, FlushError, GpuFuture};
use vulkano::VulkanLibrary;

use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::Window;

use crate::shader::vs;

#[derive(BufferContents, Vertex)]
#[repr(C)]
pub struct CustomVertex {
    #[format(R32G32B32_SFLOAT)]
    pub position: [f32; 3],
    #[format(R32G32B32_SFLOAT)]
    pub normal: [f32; 3],
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

pub fn create_swapchain(
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

fn get_standard_memory_allocator(device: Arc<Device>) -> StandardMemoryAllocator {
    StandardMemoryAllocator::new_default(device.clone())
}

pub fn create_vertex_buffer(
    device: Arc<Device>,
    vertices: Vec<CustomVertex>,
) -> Subbuffer<[CustomVertex]> {
    Buffer::from_iter(
        &get_standard_memory_allocator(device.clone()),
        BufferCreateInfo {
            usage: BufferUsage::VERTEX_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            usage: MemoryUsage::Upload,
            ..Default::default()
        },
        vertices,
    )
    .unwrap()
}

pub fn create_index_buffer(device: Arc<Device>, indices: Vec<u16>) -> Subbuffer<[u16]> {
    Buffer::from_iter(
        &get_standard_memory_allocator(device.clone()),
        BufferCreateInfo {
            usage: BufferUsage::INDEX_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            usage: MemoryUsage::Upload,
            ..Default::default()
        },
        indices,
    )
    .unwrap()
}

pub fn get_render_pass(device: Arc<Device>, swapchain: &Arc<Swapchain>) -> Arc<RenderPass> {
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

fn get_depth_buffer(device: Arc<Device>, dimensions: [u32; 2]) -> Arc<ImageView<AttachmentImage>> {
    ImageView::new_default(
        AttachmentImage::transient(
            &get_standard_memory_allocator(device),
            dimensions,
            Format::D16_UNORM,
        )
        .unwrap(),
    )
    .unwrap()
}

pub fn get_framebuffers(
    device: Arc<Device>,
    images: &[Arc<SwapchainImage>],
    render_pass: &Arc<RenderPass>,
) -> Vec<Arc<Framebuffer>> {
    images
        .iter()
        .map(|image| {
            Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![
                        get_color_buffer(image.clone()),
                        get_depth_buffer(device.clone(), images[0].dimensions().width_height())
                            .clone(),
                    ],
                    ..Default::default()
                },
            )
            .unwrap()
        })
        .collect::<Vec<_>>()
}

pub fn get_pipeline(
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

pub fn get_command_buffer(
    device: Arc<Device>,
    queue: &Arc<Queue>,
    pipeline: &Arc<GraphicsPipeline>,
    framebuffer: &Arc<Framebuffer>,
    vertex_buffer: &Subbuffer<[CustomVertex]>,
    index_buffer: &Subbuffer<[u16]>,
    set: &Arc<PersistentDescriptorSet>,
) -> Arc<PrimaryAutoCommandBuffer> {
    let command_buffer_allocator =
        StandardCommandBufferAllocator::new(device.clone(), Default::default());
    let mut builder = AutoCommandBufferBuilder::primary(
        &command_buffer_allocator,
        queue.queue_family_index(),
        CommandBufferUsage::MultipleSubmit,
    )
    .unwrap();

    builder
        .begin_render_pass(
            RenderPassBeginInfo {
                clear_values: vec![Some([0.63, 0.82, 0.96, 1.0].into()), Some(1f32.into())],
                ..RenderPassBeginInfo::framebuffer(framebuffer.clone())
            },
            SubpassContents::Inline,
        )
        .unwrap()
        .bind_pipeline_graphics(pipeline.clone())
        .bind_descriptor_sets(
            PipelineBindPoint::Graphics,
            pipeline.layout().clone(),
            0,
            set.clone(),
        )
        .bind_vertex_buffers(0, vertex_buffer.clone())
        .bind_index_buffer(index_buffer.clone())
        .draw_indexed(index_buffer.len() as u32, 1, 0, 0, 0)
        .unwrap()
        .end_render_pass()
        .unwrap();

    Arc::new(builder.build().unwrap())
}

fn get_window(surface: &Arc<Surface>) -> Arc<Window> {
    surface
        .object()
        .unwrap()
        .clone()
        .downcast::<Window>()
        .unwrap()
}

pub fn get_viewport() -> Viewport {
    Viewport {
        origin: [0.0, 0.0],
        dimensions: [1024.0, 1024.0],
        depth_range: 0.0..1.0,
    }
}

pub fn run_event_loop(
    event_loop: EventLoop<()>,
    mut swapchain: Arc<Swapchain>,
    surface: Arc<Surface>,
    render_pass: Arc<RenderPass>,
    mut viewport: Viewport,
    device: Arc<Device>,
    vertex_shader: Arc<ShaderModule>,
    fragment_shader: Arc<ShaderModule>,
    queue: Arc<Queue>,
    vertex_buffer: Subbuffer<[CustomVertex]>,
    index_buffer: Subbuffer<[u16]>,
    mut pipeline: Arc<GraphicsPipeline>,
    mut framebuffers: Vec<Arc<Framebuffer>>,
) {
    let mut recreate_swapchain = false;
    let mut previous_frame_end = Some(sync::now(device.clone()).boxed());

    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent {
            event: WindowEvent::CloseRequested,
            ..
        } => {
            *control_flow = ControlFlow::Exit;
        }
        Event::WindowEvent {
            event: WindowEvent::Resized(_),
            ..
        } => {
            recreate_swapchain = true;
        }
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
                framebuffers = get_framebuffers(device.clone(), &new_images, &render_pass);

                viewport.dimensions = get_window(&surface).inner_size().into();
                pipeline = get_pipeline(
                    device.clone(),
                    vertex_shader.clone(),
                    fragment_shader.clone(),
                    render_pass.clone(),
                    viewport.clone(),
                );
            }

            let uniform_buffer_subbuffer = {
                let aspect_ratio =
                    swapchain.image_extent()[0] as f32 / swapchain.image_extent()[1] as f32;
                let proj = cgmath::perspective(
                    Rad(std::f32::consts::FRAC_PI_2),
                    aspect_ratio,
                    0.1,
                    1000.0,
                );
                let view = Matrix4::look_at_rh(
                    Point3::new(0.0, 0.0, 2.0),
                    Point3::new(0.0, 0.0, 0.0),
                    Vector3::new(0.0, -1.0, 0.0),
                );
                let scale = Matrix4::from_scale(1.0);
                let uniform_buffer = SubbufferAllocator::new(
                    get_standard_memory_allocator(device.clone()),
                    SubbufferAllocatorCreateInfo {
                        buffer_usage: BufferUsage::UNIFORM_BUFFER,
                        ..Default::default()
                    },
                );
                let uniform_data = vs::UniformBufferObject {
                    model: Matrix4::from_angle_x(Rad(0 as f32)).into(),
                    view: (view * scale).into(),
                    projection: proj.into(),
                };
                let subbuffer = uniform_buffer.allocate_sized().unwrap();
                *subbuffer.write().unwrap() = uniform_data;
                subbuffer
            };

            let layout = pipeline.layout().set_layouts().get(0).unwrap();
            let descriptor_set_allocator = StandardDescriptorSetAllocator::new(device.clone());
            let set = PersistentDescriptorSet::new(
                &descriptor_set_allocator,
                layout.clone(),
                [WriteDescriptorSet::buffer(0, uniform_buffer_subbuffer)],
            )
            .unwrap();

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

            let command_buffer = get_command_buffer(
                device.clone(),
                &queue,
                &pipeline,
                &framebuffers[image_i as usize],
                &vertex_buffer,
                &index_buffer,
                &set.clone(),
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
