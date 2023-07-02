# 3D Model Viewer in Vulkano

## Overview

This repository contains a basic 3D model viewer implemented in Rust using the Vulkano library.

<img src="https://github.com/MarcioCerqueira/vulkano_3d_model_viewer/blob/main/erato.jpeg"  width="50%" height="50%">

## Instructions

To run the application, please run `cargo run configs\<config_file>.json`

## Scene Configuration

By default, this project comes with the [Viking Room](https://sketchfab.com/3d-models/viking-room-a49f1b8e4f5c4ecf9e1fe7d81915ad38) model by [nigeloh](https://sketchfab.com/nigelgoh) updated by the [Vulkan tutorial](https://vulkan-tutorial.com/Loading_models). 
If you want to load your own models in the application, please keep in mind that the current version of the application only supports the loading of a single .obj file (ideally stored in the `models` folder), associated with a single texture (ideally stored in the `textures` folder), that already has precomputed normals (`vn` items). 
Then, create a configuration file following the examples available in the `configs` folder to setup the scene to be loaded.

This application was successfully tested using the following models:
- [Dabrovic Sponza](https://casual-effects.com/g3d/data10/research/model/dabrovic_sponza/sponza.zip), after exporting a new .obj file with precomputed normals using [3D Viewer](https://3dviewer.net/);
- [Erato](https://casual-effects.com/g3d/data10/research/model/erato/erato.zip);
- [Viking Room](https://vulkan-tutorial.com/Loading_models);

Tips: If you want to set up the camera properly, you can move the camera around the scene and press `C` to capture the camera data and later insert it in the configuration file.
