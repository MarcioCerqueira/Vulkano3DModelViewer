#version 460

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
            
layout(location = 0) out vec3 fragNormal;

layout(set = 0, binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 projection;
} ubo;

void main() {
    gl_Position = ubo.projection * ubo.view * ubo.model * vec4(position, 1.0);
    fragNormal = mat3(transpose(inverse(ubo.model))) * normal;
}