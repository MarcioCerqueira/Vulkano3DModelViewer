#version 460

layout(set = 0, binding = 1) uniform sampler2D texSampler;

layout(location = 0) in vec3 fragPosition;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec3 fragTextureCoords;
layout(location = 3) in vec3 cameraPosition;

layout(location = 0) out vec4 outColor;

float computeAmbientLightIntensity() 
{
    return 0.5;
}       

float computeDiffuseLightIntensity(vec3 normal, vec3 lightDirection)
{
    return 0.75 * max(dot(normal, lightDirection), 0.0);
}

float computeSpecularLightIntensity(vec3 normal, vec3 lightDirection)
{
    vec3 viewDirection = normalize(cameraPosition - fragPosition);
    vec3 reflectDirection = reflect(-lightDirection, normal);
    return 0.75 * pow(max(dot(viewDirection, reflectDirection), 0.0), 0.5);
}

void main() 
{
    vec3 normal = normalize(fragNormal);
    vec3 lightDirection = normalize(vec3(1.0, 1.0, 1.0));
    float ambientLightIntensity = computeAmbientLightIntensity();
    float diffuseLightIntensity = computeDiffuseLightIntensity(normal, lightDirection);
    float specularLightIntensity = computeSpecularLightIntensity(normal, lightDirection);
    float totalLightIntensity = ambientLightIntensity + diffuseLightIntensity + specularLightIntensity;
    vec4 originalFragmentColor = texture(texSampler, fragTextureCoords.xy);
    outColor = totalLightIntensity * originalFragmentColor;
}