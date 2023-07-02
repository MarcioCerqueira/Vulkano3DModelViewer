use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize)]
pub struct Window {
    pub width: u32,
    pub height: u32,
}

#[derive(Deserialize, Serialize)]
pub struct Vector3D {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

#[derive(Deserialize, Serialize)]
pub struct Camera {
    pub position: Vector3D,
    pub target: Vector3D,
    pub up: Vector3D,
}

#[derive(Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Scene {
    pub model_filename: String,
    pub texture_filename: String,
    pub camera: Camera,
}

#[derive(Deserialize, Serialize)]
pub struct Config {
    pub window: Window,
    pub scene: Scene,
}

pub fn read_file(config_filename: &str) -> Config {
    let config = std::fs::read_to_string(config_filename).unwrap();
    serde_json::from_str::<Config>(&config).unwrap()
}
