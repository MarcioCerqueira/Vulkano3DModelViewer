#version 460

layout(set = 0, binding = 1) uniform sampler2D texSampler;

layout(location = 0) in vec3 fragNormal;
layout(location = 1) in vec3 fragTextureCoords;

layout(location = 0) out vec4 outColor;

float computeAmbientLightIntensity() 
{
    return 0.5;
}       

float computeDiffuseLightIntensity(vec3 normal, vec3 lightDirection)
{
    return 0.75 * max(dot(normal, lightDirection), 0.0);
}

void main() 
{
    vec3 normal = normalize(fragNormal);
    vec3 lightDirection = normalize(vec3(1.0, 1.0, 1.0));
    float ambientLightIntensity = computeAmbientLightIntensity();
    float diffuseLightIntensity = computeDiffuseLightIntensity(normal, lightDirection);
    float totalLightIntensity = ambientLightIntensity + diffuseLightIntensity;
    vec4 originalFragmentColor = texture(texSampler, fragTextureCoords.xy);
    outColor = totalLightIntensity * originalFragmentColor;
}