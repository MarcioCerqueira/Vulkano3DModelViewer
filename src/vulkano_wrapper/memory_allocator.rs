use std::sync::Arc;
use vulkano::buffer::allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo};
use vulkano::buffer::BufferUsage;
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::device::Device;
use vulkano::memory::allocator::StandardMemoryAllocator;

pub struct MemoryAllocator {
    pub command_buffer: Arc<StandardCommandBufferAllocator>,
    pub descriptor_set: Arc<StandardDescriptorSetAllocator>,
    pub standard: Arc<StandardMemoryAllocator>,
    pub subbuffer: Arc<SubbufferAllocator>,
}

pub fn new(device: Arc<Device>) -> MemoryAllocator {
    MemoryAllocator {
        command_buffer: Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            Default::default(),
        )),
        descriptor_set: Arc::new(StandardDescriptorSetAllocator::new(device.clone())),
        standard: Arc::new(StandardMemoryAllocator::new_default(device.clone())),
        subbuffer: Arc::new(SubbufferAllocator::new(
            Arc::new(StandardMemoryAllocator::new_default(device.clone())),
            SubbufferAllocatorCreateInfo {
                buffer_usage: BufferUsage::UNIFORM_BUFFER,
                ..Default::default()
            },
        )),
    }
}
