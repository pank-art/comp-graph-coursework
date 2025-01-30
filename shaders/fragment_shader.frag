#version 330 core

in vec3 position;
in vec3 normals_sphere;   // flat in vec3 normals;
flat in vec3 normals_cube;
in vec2 fragTextureCoord;  // Координаты текстуры

out vec4 fragColor;

uniform float morphFactor;
uniform vec3 lightPosition;
uniform vec3 lightColor;
uniform vec3 ambientColor;
uniform vec3 diffuseColor;
uniform vec3 specularColor;
uniform float shininess;
uniform sampler2D textureSampler;  // Сэмплер текстуры
uniform int tex_enable;

void main()
{
    vec3 normal = morphFactor == 0.0 ? normalize(normals_cube) : normalize(normals_sphere);

    // Рассчитываем вектор направления света
    vec3 lightDirection = normalize(lightPosition - position);

    // Рассчитываем фоновое освещение
    vec3 ambient = ambientColor * lightColor;

    // Рассчитываем диффузное освещение
    float diffuseFactor = max(dot(normal, lightDirection), 0.0);
    vec3 diffuse = diffuseColor * lightColor * diffuseFactor;

    // Рассчитываем зеркальное отражение
    vec3 viewDirection = normalize(-position);
    vec3 reflectDirection = reflect(-lightDirection, normal);
    float specularFactor = pow(max(dot(viewDirection, reflectDirection), 0.0), shininess);
    vec3 specular = specularColor * lightColor * specularFactor;


    vec4 textureColor = texture(textureSampler, fragTextureCoord);  // Получение цвета из текстуры

    // Общий цвет с учетом всех эффектов освещения
    vec3 finalColor = ambient + diffuse + specular;


    // Присваиваем цвет фрагменту
    fragColor = vec4(finalColor, 1.0) * mix(vec4(1.0), textureColor, tex_enable);
    //fragColor = vec4(normal * 0.5 + 0.5, 1.0);
    //fragColor = textureColor;
}