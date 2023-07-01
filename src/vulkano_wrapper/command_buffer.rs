use std::sync::Arc;

use vulkano::buffer::Subbuffer;
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer, RenderPassBeginInfo,
    SubpassContents,
};
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::device::{Device, Queue};
use vulkano::pipeline::{GraphicsPipeline, Pipeline, PipelineBindPoint};
use vulkano::render_pass::Framebuffer;
use vulkano::sampler::{Filter, Sampler, SamplerAddressMode, SamplerCreateInfo};

use crate::camera::Camera;
use crate::vulkano_wrapper::memory_allocator::MemoryAllocator;
use crate::vulkano_wrapper::model::VulkanoModel;
use crate::vulkano_wrapper::shader::vs;

pub fn create_command_buffer_builder(
    memory_allocator: &MemoryAllocator,
    queue: &Arc<Queue>,
) -> AutoCommandBufferBuilder<PrimaryAutoCommandBuffer, Arc<StandardCommandBufferAllocator>> {
    AutoCommandBufferBuilder::primary(
        &memory_allocator.command_buffer,
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap()
}

pub fn get_command_buffer(
    queue: &Arc<Queue>,
    pipeline: &Arc<GraphicsPipeline>,
    framebuffer: &Arc<Framebuffer>,
    vulkano_model: &VulkanoModel,
    device: &Arc<Device>,
    memory_allocator: &MemoryAllocator,
    camera: &Camera,
) -> Arc<PrimaryAutoCommandBuffer> {
    let mut builder = create_command_buffer_builder(memory_allocator, queue);
    let persistent_descriptor_set = create_persistent_descriptor_set(
        &memory_allocator,
        &pipeline,
        camera,
        &vulkano_model,
        device.clone(),
    );

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
            persistent_descriptor_set,
        )
        .bind_vertex_buffers(0, vulkano_model.vertex_buffer.clone())
        .bind_index_buffer(vulkano_model.index_buffer.clone())
        .draw_indexed(vulkano_model.index_buffer.len() as u32, 1, 0, 0, 0)
        .unwrap()
        .end_render_pass()
        .unwrap();

    Arc::new(builder.build().unwrap())
}

fn create_uniform_buffer_object(
    memory_allocator: &MemoryAllocator,
    camera: &Camera,
) -> Subbuffer<vs::UniformBufferObject> {
    let mut uniform_data = vs::UniformBufferObject {
        model: camera.get_model_matrix().into(),
        view: camera.get_view_matrix().into(),
        projection: camera.get_projection_matrix().into(),
        camera_position: camera.get_eye().into(),
    };
    uniform_data.projection[1][1] = uniform_data.projection[1][1] * -1.0;
    let subbuffer = memory_allocator.subbuffer.allocate_sized().unwrap();
    *subbuffer.write().unwrap() = uniform_data;
    subbuffer
}

fn create_sampler(device: Arc<Device>) -> Arc<Sampler> {
    Sampler::new(
        device.clone(),
        SamplerCreateInfo {
            mag_filter: Filter::Linear,
            min_filter: Filter::Linear,
            address_mode: [SamplerAddressMode::Repeat; 3],
            ..Default::default()
        },
    )
    .unwrap()
}

fn create_persistent_descriptor_set(
    memory_allocator: &MemoryAllocator,
    pipeline: &Arc<GraphicsPipeline>,
    camera: &Camera,
    vulkano_model: &VulkanoModel,
    device: Arc<Device>,
) -> Arc<PersistentDescriptorSet> {
    PersistentDescriptorSet::new(
        &memory_allocator.descriptor_set,
        pipeline.layout().set_layouts().get(0).unwrap().clone(),
        [
            WriteDescriptorSet::buffer(0, create_uniform_buffer_object(&memory_allocator, camera)),
            WriteDescriptorSet::image_view_sampler(
                1,
                vulkano_model.texture_buffer.clone(),
                create_sampler(device.clone()),
            ),
        ],
    )
    .unwrap()
}
