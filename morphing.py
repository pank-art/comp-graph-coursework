import glm
from PIL import Image
import glfw
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np

from utils import *
from window_manager import init_window, terminate
from shape import SPHERE, CUBE
from callbacks import key_callback, mouse_button_callback, cursor_position_callback, scroll_callback


WIDTH, HEIGHT = 1000, 800
t = 0.5
direction = 1
morph = True
# Координаты вершин куба
is_mouse_pressed = False
last_mouse_pos = [0, 0]
rotation_angle_x = 0.0
rotation_angle_y = 0.0
tex_enable, fill = False, False
n = 40
velocity = 0
G = 9.8
height = 1
last_time = 0
last_time2 = 0
animation = False
tex_id = 0
morph = True

vertices = np.array([
    [-0.3, -0.3, 0.3],  # Вершина 0
    [0.3, -0.3, 0.3],  # Вершина 1
    [0.3, 0.3, 0.3],  # Вершина 2
    [-0.3, 0.3, 0.3],  # Вершина 3
    [-0.3, -0.3, -0.3],  # Вершина 4
    [0.3, -0.3, -0.3],  # Вершина 5
    [0.3, 0.3, -0.3],  # Вершина 6
    [-0.3, 0.3, -0.3]  # Вершина 7
])

setInfinityDistantLight = False
alpha = 0
ambientMode = 3
ambient = [[0, 0, 0, 1], [1, 1, 1, 0.5], [1, 1, 1, 1], [0.5, 0.5, 0.5, 1], [0, 1, 0, 1]]
diffuseMode = 0
diffuse = [[1, 0.9, 0.8, 1], [0, 0, 0, 1], [1, 1, 1, 0.5], [0.5, 0.5, 0.5, 1], [0, 1, 0, 1]]
specularMode = 0
specular = [[1, 1, 1, 1], [0, 0, 0, 1], [1, 1, 1, 0.5], [0.5, 0.5, 0.5, 1], [0, 1, 0, 1]]
shininess = 128.0
light_color = [1.0, 0.9, 0.8]

# Шейдерный код вершинного шейдера
vertex_shader_code = """
#version 330 core
layout(location = 0) in vec3 cubePosition;
layout(location = 1) in vec3 cubeNormal;     // Нормали для куба
layout(location = 2) in vec3 spherePosition;   // Позиции вершин сферы

uniform float morphFactor;

out vec3 position;
out vec3 normals_cube;
out vec3 normals_sphere;

uniform mat4 model;
uniform mat4 projection;
uniform mat4 view;

void main() {
    vec3 morphedPosition = mix(cubePosition, spherePosition, morphFactor);
    position = vec3(view * model * vec4(morphedPosition, 1.0));
    
    vec3 morphedNormal = mix(cubeNormal, normalize(spherePosition), morphFactor);
    normals_cube = mat3(transpose(inverse(view * model))) * cubeNormal;;
    normals_sphere = mat3(transpose(inverse(view * model))) * morphedNormal;;
    
    gl_Position = projection * view * model * vec4(morphedPosition, 1.0);
}
"""

# Шейдерный код фрагментного шейдера
fragment_shader_code = """
#version 330 core

in vec3 position;
in vec3 normals_sphere;   // flat in vec3 normals; 
flat in vec3 normals_cube; 

out vec4 fragColor;

uniform float morphFactor;
uniform vec3 lightPosition;
uniform vec3 lightColor;
uniform vec3 ambientColor;
uniform vec3 diffuseColor;
uniform vec3 specularColor;
uniform float shininess;
uniform sampler2D textureSampler;  // Сэмплер текстуры
uniform vec2 textureCoord;
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


    vec4 textureColor = texture(textureSampler, textureCoord);  // Получение цвета из текстуры

    // Общий цвет с учетом всех эффектов освещения
    vec3 finalColor = ambient + diffuse + specular;


    // Присваиваем цвет фрагменту
    fragColor = vec4(finalColor, 1.0) * mix(vec4(1.0), textureColor, tex_enable);
    //fragColor = vec4(normals * 0.5 + 0.5, 1.0);
    //fragColor = textureColor;
}

"""


def draw_interpolated_object(t, vbo_vertices, vbo_normals, cube_vertices, sphere_v):
    interpolated = cube.interpolate_vertices(t, cube_vertices, sphere_v)
    normals = update_normals(interpolated)
    if t < 0.5:
        normals = update_normals(interpolated)
    else:
        normals = calculate_normals_for_sphere(interpolated)

    glBindBuffer(GL_ARRAY_BUFFER, vbo_vertices)
    glBufferSubData(GL_ARRAY_BUFFER, 0, interpolated.nbytes, interpolated)
    glBindBuffer(GL_ARRAY_BUFFER, 0)

    glBindBuffer(GL_ARRAY_BUFFER, vbo_normals)
    glBufferSubData(GL_ARRAY_BUFFER, 0, normals.nbytes, normals)
    glBindBuffer(GL_ARRAY_BUFFER, 0)


def rotation_matrix(i, f):
    i = np.asarray(i)
    assert i.size == 3, "i must be a 3d vector"
    # Normalize i
    i /= np.linalg.norm(i)

    c, s = np.cos(f), np.sin(f)
    a = 1 - c

    # Build the rotation matrix
    R = np.array([[i[0] ** 2 * a + c, i[0] * i[1] * a - i[2] * s, i[0] * i[2] * a + i[1] * s, 0],
                  [i[0] * i[1] * a + i[2] * s, i[1] ** 2 * a + c, i[1] * i[2] * a - i[0] * s, 0],
                  [i[0] * i[2] * a - i[1] * s, i[1] * i[2] * a + i[0] * s, i[2] ** 2 * a + c, 0],
                  [0, 0, 0, 1]], dtype=np.float32)
    return R


def setLight(shader_program):
    light_position = [5, 5, 5, 1]

    light_position_location = glGetUniformLocation(shader_program, "lightPosition")
    light_color_location = glGetUniformLocation(shader_program, "lightColor")
    ambient_color_location = glGetUniformLocation(shader_program, "ambientColor")
    diffuse_color_location = glGetUniformLocation(shader_program, "diffuseColor")
    specular_color_location = glGetUniformLocation(shader_program, "specularColor")
    shininess_location = glGetUniformLocation(shader_program, "shininess")

    glUniform3fv(light_position_location, 1, light_position)
    glUniform3fv(light_color_location, 1, light_color)
    glUniform3fv(ambient_color_location, 1, ambient[ambientMode])
    glUniform3fv(diffuse_color_location, 1, diffuse[diffuseMode])
    glUniform3fv(specular_color_location, 1, specular[specularMode])
    glUniform1f(shininess_location, shininess)


def make_texture(filename):
    img = Image.open(filename)
    img_data = np.array(list(img.getdata()), dtype=np.uint8)

    # Генерация и привязка текстуры
    texture_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture_id)

    # Загрузка данных текстуры
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img.size[0], img.size[1], 0,
                 GL_RGB, GL_UNSIGNED_BYTE, img_data)

    # Настройка параметров фильтрации и повторения текстуры
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)

    # Возвращаем идентификатор текстуры
    # return texture_id


def find_points():
    points = np.zeros((n + 1, n + 1, 3), dtype=np.float32)
    # ищем точки эллипса
    for i in range(n + 1):
        for j in range(n + 1):
            ang2 = 2 * math.pi * i / n
            ang = 2 * math.pi * j / n + math.pi / 2
            z = 0.3 * math.cos(ang) * math.cos(ang2)
            y = 0.3 * math.cos(ang) * math.sin(ang2)
            x = 1.0 * math.sin(ang)
            points[i][j] = x, y, z
    return points


def dot(v1, v2):
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]


def update_normals(vertices):
    normals = np.zeros_like(vertices)
    n = len(vertices)
    num_faces = 6
    missing_points = n % num_faces
    points_in_faces = []
    for face_index in range(num_faces):
        points = n // num_faces
        if face_index < missing_points % num_faces:
            points += 1  # Распределяем остаток равномерно

        points += points_in_faces[face_index-1] if face_index != 0 else 0
        points_in_faces.append(points)

    no_start_tr = set()
    for i in points_in_faces:
        no_start_tr.add(i - 1)
        no_start_tr.add(i - 2)
    print(no_start_tr)
    # Проходим по треугольникам (каждое треугольное лицо состоит из 3 индексов)
    for i in range(0, len(vertices) - 2, 1):
        if i in no_start_tr:
            continue
        i1, i2, i3 = i, i + 1, i + 2

        # Получаем вершины треугольника
        v1 = vertices[i1]
        v2 = vertices[i2]
        v3 = vertices[i3]

        # Вычисляем векторы сторон
        edge1 = v2 - v1
        edge2 = v3 - v1

        # Нормаль треугольника — векторное произведение
        normal = np.cross(edge1, edge2)

        if np.linalg.norm(normal) != 0:
            # Присваиваем нормаль каждой вершине треугольника
            normals[i1] += normal
            normals[i2] += normal
            normals[i3] += normal

            normals[i1] = normals[i1] / np.linalg.norm(normals[i1])
            normals[i2] = normals[i2] / np.linalg.norm(normals[i2])
            normals[i3] = normals[i3] / np.linalg.norm(normals[i3])
        else:
            print(i, normals[i])
            normals[i2] = normals[i1]
            normals[i3] = normals[i1]

    # Нормализуем все нормали
    normals = normals / np.linalg.norm(normals, axis=1)[:, np.newaxis]

    # Ориентируем нормали наружу (проверяем их ориентацию относительно внешнего вектора)
    # Для этого используем точку внутри объекта (например, центр) и смотрим на угол с нормалью
    center = np.mean(vertices, axis=0)  # Центральная точка объекта

    for i in range(len(vertices)):
        # Направление от центра к вершине
        direction_to_center = vertices[i] - center
        # Скалярное произведение с нормалью
        if np.dot(normals[i], direction_to_center) < 0:
            # Если нормаль направлена внутрь (отрицательное скалярное произведение), инвертируем её
            normals[i] = -normals[i]

    # for x in no_start_tr:
    #     print(normals[x])


    return normals


def calculate_normals_for_sphere(vertices):
    """
    Вычисляет нормали для сферической поверхности,
    предполагая, что нормаль каждой вершины совпадает с направлением от центра сферы.

    :param vertices: numpy.ndarray, массив с координатами вершин (N, 3)
    :return: numpy.ndarray, массив нормалей для каждой вершины (N, 3)
    """
    # Нормализуем каждую вершину для получения направления нормали
    normals = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
    return normals


def look_at(eye, target, up):
    f = (target - eye)
    f = f / np.linalg.norm(f)

    u = up / np.linalg.norm(up)
    s = np.cross(f, u)
    u = np.cross(s, f)

    m = np.eye(4, dtype=np.float32)
    m[0, :3] = s
    m[1, :3] = u
    m[2, :3] = -f
    m[:3, 3] = -np.dot(m[:3, :3], eye)

    return m.T


def display(window, shader_program, vao):
    global t, direction
    global height, velocity, last_time, last_time2
    glUseProgram(shader_program)
    glBindVertexArray(vao)

    # текстура и свет
    if tex_enable:
        glUniform1i(glGetUniformLocation(shader_program, "tex_enable"), 1)
    else:
        glUniform1i(glGetUniformLocation(shader_program, "tex_enable"), 0)

    setLight(shader_program)

    # Очистка буфера цвета
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    # Отрисовка куба
    # Пример построения матрицы вращения
    v_x = np.array([1.0, 0.0, 0.0])  # Ось X
    v_y = np.array([0.0, 1.0, 0.0])  # Ось Y

    rotation_x = rotation_matrix(v_x, np.radians(rotation_angle_x))
    rotation_y = rotation_matrix(v_y, np.radians(rotation_angle_y))

    model = rotation_x @ rotation_y

    time = glutGet(GLUT_ELAPSED_TIME)
    if time - last_time > 5:
        if height - velocity > 0:
            height -= velocity
            if velocity < 0 < velocity + G / 100000:
                velocity = 0
            else:
                velocity += G / 100000
        else:
            height = 0
            velocity = -velocity
        last_time = time

    if not animation:
        model = model @ np.array([[1, 0, 0, 0],
                                  [0, 1, 0, 0],
                                  [0, 0, 1, 0],
                                  [0, 0, 0, 1]])
    else:
        model = model @ np.array([[1, 0, 0, 0],
                                  [0, 1, 0, 0],
                                  [0, 0, 1, 0],
                                  [0, height, 0, 1]])

    glUniformMatrix4fv(glGetUniformLocation(shader_program, "model"), 1, GL_FALSE, model)

    EPSILON = 1e-6  # Погрешность для проверки границ
    PAUSE_TIME = 5000  # Время задержки на границах в миллисекундах

    if morph:
        current_time = time  # Текущее время для удобства

        # Проверка: задерживаем t на границах
        if t >= 1.0 - EPSILON:
            t = 1.0
            if current_time - last_time2 > PAUSE_TIME:
                if direction > 0:
                    direction *= -1  # Меняем направление движения
                last_time2 = current_time  # Обновляем время последней паузы
                t += 0.001 * direction
        elif t <= 0.0 + EPSILON:
            t = 0.0
            if current_time - last_time2 > PAUSE_TIME:
                if direction < 0:
                    direction *= -1  # Меняем направление движения
                last_time2 = current_time  # Обновляем время последней паузы
                t += 0.001 * direction
        else:
            # t движется непрерывно внутри диапазона (0, 1)
            t += 0.001 * direction
            last_time2 = current_time

    #draw_interpolated_object(t, vbo_vertices, vbo_normals, cube_vertices, sphere_v)  # Draw the cube
    morph_factor_location = glGetUniformLocation(shader_program, "morphFactor")
    glUniform1f(morph_factor_location, t)
    print(t)
    glDrawArrays(GL_TRIANGLE_STRIP, 0, n)

    if fill:  # switching between wireframe and solid-state model display
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
    else:
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

    # Swap the front and back buffers
    glfw.swap_buffers(window)
    # Poll for and process events
    glfw.poll_events()


def main():
    global tex_id, n
    # Инициализация библиотеки GLFW
    if not glfw.init():
        return
    # Set GLFW window options
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.RESIZABLE, False)
    # Устанавливаем параметры для окна перед его созданием
    glfw.window_hint(glfw.DEPTH_BITS, 24)  # Задаём 24 бита для глубины
    # Создание окна
    window = glfw.create_window(WIDTH, HEIGHT, "Cube", None, None)
    if not window:
        glfw.terminate()
        return

    # Установка контекста OpenGL для окна
    glfw.make_context_current(window)
    glfw.set_key_callback(window, key_callback)
    glfw.set_mouse_button_callback(window, mouse_button_callback)
    glfw.set_cursor_pos_callback(window, cursor_position_callback)

    glViewport(0, 0, WIDTH, HEIGHT)
    glEnable(GL_DEPTH_TEST)

    # Компиляция и связывание шейдеров
    shader_program = compileProgram(
        compileShader(vertex_shader_code, GL_VERTEX_SHADER),
        compileShader(fragment_shader_code, GL_FRAGMENT_SHADER)
    )
    glUseProgram(shader_program)

    # Создание буфера вершин
    sphere = SPHERE()
    n = len(sphere.vertices)
    cube = CUBE(n)
    sphere_v = sphere.vertices
    all_points = cube.vertices
    normals = cube.vertices
    n = len(all_points)

    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)

    vbo_cube = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo_cube)
    glBufferData(GL_ARRAY_BUFFER, all_points.nbytes, all_points, GL_STATIC_DRAW)

    vbo_sphere = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo_sphere)
    glBufferData(GL_ARRAY_BUFFER, sphere_v.nbytes, sphere_v, GL_STATIC_DRAW)

    vbo_normals = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo_normals)
    glBufferData(GL_ARRAY_BUFFER, normals.nbytes, normals, GL_STATIC_DRAW)

    # Получение ссылки на атрибут позиции вершин в шейдере
    position_attrib = glGetAttribLocation(shader_program, "cubePosition")
    glBindBuffer(GL_ARRAY_BUFFER, vbo_cube)
    glVertexAttribPointer(position_attrib, 3, GL_FLOAT, GL_FALSE, 0, None)
    glEnableVertexAttribArray(position_attrib)

    position_sphere = glGetAttribLocation(shader_program, "spherePosition")
    glBindBuffer(GL_ARRAY_BUFFER, vbo_sphere)
    glVertexAttribPointer(position_sphere, 3, GL_FLOAT, GL_FALSE, 0, None)
    glEnableVertexAttribArray(position_sphere)

    normal_attrib = glGetAttribLocation(shader_program, "cubeNormal")
    glBindBuffer(GL_ARRAY_BUFFER, vbo_normals)
    glVertexAttribPointer(normal_attrib, 3, GL_FLOAT, GL_FALSE, 0, None)
    glEnableVertexAttribArray(normal_attrib)

    projection_location = glGetUniformLocation(shader_program, "projection")
    # Установка матрицы проекции
    projection = glm.perspective(glm.radians(45.0), WIDTH / HEIGHT, 0.1, 100.0)
    glUniformMatrix4fv(projection_location, 1, GL_FALSE, glm.value_ptr(projection))

    camera_position = np.array([1, 1, -2])  # Камера отодвинута назад
    camera_target = np.array([0, 0, 0])  # Центр фигуры
    upp = np.array([0, 1, 0])  # Направление "вверх"
    view = look_at(camera_position, camera_target, upp)
    view_location = glGetUniformLocation(shader_program, "view")
    glUniformMatrix4fv(view_location, 1, GL_FALSE, view)

    make_texture('textures/texture.bmp')
    glUniform2fv(glGetUniformLocation(shader_program, "textureCoord"), 1, [0, 1])
    glUniform1i(glGetUniformLocation(shader_program, "textureSampler"), 0)

    glBindVertexArray(0)

    # Основной цикл программы
    while not glfw.window_should_close(window):
        display(window, shader_program, vao)

        # Освобождение ресурсов
    glDeleteBuffers(1, [vbo_cube])
    glDeleteBuffers(1, [vbo_normals])
    glDeleteBuffers(1, [vbo_sphere])
    glDeleteProgram(shader_program)

    # Закрытие окна и завершение работы
    glfw.terminate()






if __name__ == "__main__":
    main()
