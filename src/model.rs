use obj::{load_obj, Obj};
use std::fs::File;
use std::io::BufReader;

use crate::vulkano_wrapper::CustomVertex;

pub struct Model {
    pub vertices: Vec<CustomVertex>,
    pub indices: Vec<u16>,
}

impl Model {
    pub fn new(filename: &str) -> Model {
        let file = BufReader::new(File::open(filename).expect("Failed to read model!"));
        let mesh: Obj = load_obj(file).expect("Failed to load OBJ file");
        Model {
            vertices: mesh
                .vertices
                .iter()
                .map(|vertex| CustomVertex {
                    position: vertex.position,
                    normal: vertex.normal,
                })
                .collect(),
            indices: mesh.indices,
        }
    }
}
