use std::sync::Arc;
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::device::Device;
use vulkano::memory::allocator::StandardMemoryAllocator;

pub struct MemoryAllocator {
    pub command_buffer: Arc<StandardCommandBufferAllocator>,
    pub descriptor_set: Arc<StandardDescriptorSetAllocator>,
    pub standard: Arc<StandardMemoryAllocator>,
}

pub fn new(device: Arc<Device>) -> MemoryAllocator {
    MemoryAllocator {
        command_buffer: Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            Default::default(),
        )),
        descriptor_set: Arc::new(StandardDescriptorSetAllocator::new(device.clone())),
        standard: Arc::new(StandardMemoryAllocator::new_default(device.clone())),
    }
}
