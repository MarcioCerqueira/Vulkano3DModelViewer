use crate::vulkano_wrapper::CustomVertex;
pub struct Model {
    pub vertices: Vec<CustomVertex>,
}

impl Model {
    pub fn new() -> Model {
        let vertex1 = CustomVertex {
            position: [-0.5, -0.5],
        };
        let vertex2 = CustomVertex {
            position: [0.0, 0.5],
        };
        let vertex3 = CustomVertex {
            position: [0.5, -0.25],
        };
        Model {
            vertices: vec![vertex1, vertex2, vertex3],
        }
    }
}
