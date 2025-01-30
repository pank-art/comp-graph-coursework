#version 330 core
layout(location = 0) in vec3 cubePosition;
layout(location = 1) in vec3 cubeNormal;     // Нормали для куба
layout(location = 2) in vec3 spherePosition;   // Позиции вершин сферы
layout(location = 3) in vec2 textureCube;
layout(location = 4) in vec2 textureSphere;

uniform float morphFactor;

out vec3 position;
out vec3 normals_cube;
out vec3 normals_sphere;
out vec2 fragTextureCoord;

uniform mat4 model;
uniform mat4 projection;
uniform mat4 view;

void main() {
    fragTextureCoord = mix(textureCube, textureSphere, morphFactor);

    vec3 morphedPosition = mix(cubePosition, spherePosition, morphFactor);
    position = vec3(view * model * vec4(morphedPosition, 1.0));

    vec3 morphedNormal = mix(cubeNormal, normalize(spherePosition), morphFactor);
    normals_cube = mat3(transpose(inverse(view * model))) * cubeNormal;
    normals_sphere = mat3(transpose(inverse(view * model))) * morphedNormal;

    gl_Position = projection * view * model * vec4(morphedPosition, 1.0);
}