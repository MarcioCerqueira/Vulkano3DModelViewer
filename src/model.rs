use image::io::Reader as ImageReader;
use image::DynamicImage;
use obj::{load_obj, Obj, TexturedVertex};
use std::fs::File;
use std::io::BufReader;

use crate::vulkano_wrapper::CustomVertex;

pub struct Model {
    pub vertices: Vec<CustomVertex>,
    pub indices: Vec<u16>,
    pub texture: DynamicImage,
}

impl Model {
    pub fn new(object_filename: &str, texture_filename: &str) -> Model {
        let file = BufReader::new(File::open(object_filename).expect("Failed to read model!"));
        let mesh: Obj<TexturedVertex> = load_obj(file).expect("Failed to load OBJ file");

        Model {
            vertices: mesh
                .vertices
                .iter()
                .map(|vertex| CustomVertex {
                    position: vertex.position,
                    normal: vertex.normal,
                    texture_coords: vertex.texture,
                })
                .collect(),
            indices: mesh.indices,
            texture: ImageReader::open(texture_filename)
                .unwrap()
                .decode()
                .unwrap(),
        }
    }
}
