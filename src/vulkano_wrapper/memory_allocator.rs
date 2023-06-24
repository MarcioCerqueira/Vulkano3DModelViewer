use std::sync::Arc;
use vulkano::device::Device;
use vulkano::memory::allocator::StandardMemoryAllocator;

pub struct MemoryAllocator {
    pub standard: Arc<StandardMemoryAllocator>,
}

pub fn new(device: Arc<Device>) -> MemoryAllocator {
    MemoryAllocator {
        standard: Arc::new(StandardMemoryAllocator::new_default(device.clone())),
    }
}
