use crate::model::Model;
use crate::vulkano_wrapper::CustomVertex;
use crate::vulkano_wrapper::MemoryAllocator;
use image::DynamicImage;
use std::sync::Arc;
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer};
use vulkano::format::Format;
use vulkano::image::view::ImageView;
use vulkano::image::{ImageDimensions, ImmutableImage, MipmapsCount};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryUsage};

pub struct VulkanoModel {
    pub vertex_buffer: Subbuffer<[CustomVertex]>,
    pub index_buffer: Subbuffer<[u16]>,
    pub texture_buffer: Arc<ImageView<ImmutableImage>>,
}

fn create_vertex_buffer(
    vertices: Vec<CustomVertex>,
    memory_allocator: &MemoryAllocator,
) -> Subbuffer<[CustomVertex]> {
    Buffer::from_iter(
        &memory_allocator.standard,
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

fn create_index_buffer(indices: Vec<u16>, memory_allocator: &MemoryAllocator) -> Subbuffer<[u16]> {
    Buffer::from_iter(
        &memory_allocator.standard,
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

fn create_texture_buffer(
    texture: DynamicImage,
    memory_allocator: &MemoryAllocator,
    mut builder: &mut AutoCommandBufferBuilder<
        PrimaryAutoCommandBuffer,
        Arc<StandardCommandBufferAllocator>,
    >,
) -> Arc<ImageView<ImmutableImage>> {
    let dimensions = ImageDimensions::Dim2d {
        width: texture.width(),
        height: texture.height(),
        array_layers: 1,
    };
    let image = ImmutableImage::from_iter(
        &memory_allocator.standard,
        texture.clone().into_rgba8().into_raw().clone(),
        dimensions,
        MipmapsCount::One,
        Format::R8G8B8A8_SRGB,
        &mut builder,
    )
    .unwrap();
    ImageView::new_default(image).unwrap()
}

impl VulkanoModel {
    pub fn new(
        model: Model,
        memory_allocator: &MemoryAllocator,
        builder: &mut AutoCommandBufferBuilder<
            PrimaryAutoCommandBuffer,
            Arc<StandardCommandBufferAllocator>,
        >,
    ) -> VulkanoModel {
        VulkanoModel {
            vertex_buffer: create_vertex_buffer(model.vertices, memory_allocator),
            index_buffer: create_index_buffer(model.indices, memory_allocator),
            texture_buffer: create_texture_buffer(model.texture, memory_allocator, builder),
        }
    }
}
