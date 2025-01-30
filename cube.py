import glfw
from OpenGL.GL import *
from OpenGL.GLUT import *
import numpy as np
import math
import glm

import morphing
import shape
from shape import *

angle = 0.0
angle_2 = 0.0
size = 0.0
switch = True

t = 0.0
direction = 1
morph = True





colors = [
    [1.0, 0.0, 0.0],  # Красный
    [0.0, 1.0, 0.0],  # Зеленый
    [0.0, 0.0, 1.0],  # Синий
    [1.0, 1.0, 0.0],  # Желтый
    [1.0, 0.0, 1.0],  # Фиолетовый
    [0.0, 1.0, 1.0]  # Голубой
]

setInfinityDistantLight = False
alpha = 0
ambientMode = 0
ambient = [[0, 0, 0, 1], [1, 1, 1, 0.5], [1, 1, 1, 1], [0.5, 0.5, 0.5, 1], [0, 1, 0, 1]]
diffuseMode = 0
diffuse = [[1, 1, 1, 1], [0, 0, 0, 1], [1, 1, 1, 0.5], [0.5, 0.5, 0.5, 1], [0, 1, 0, 1]]
specularMode = 0
specular = [[1, 1, 1, 1], [0, 0, 0, 1], [1, 1, 1, 0.5], [0.5, 0.5, 0.5, 1], [0, 1, 0, 1]]

# Шейдеры
VERTEX_SHADER_SOURCE = """
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;

out vec3 FragPos;
out vec3 Normal;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main() {
    FragPos = vec3(model * vec4(aPos, 1.0));
    Normal = mat3(transpose(inverse(model))) * aNormal;
    gl_Position = projection * view * vec4(FragPos, 1.0);
}
"""

FRAGMENT_SHADER_SOURCE = """
#version 330 core
out vec4 FragColor;

in vec3 FragPos;
in vec3 Normal;

uniform vec3 lightPos;
uniform vec3 viewPos;
uniform vec3 lightColor;
uniform vec3 objectColor;

void main() {
    // Ambient lighting
    float ambientStrength = 0.1;
    vec3 ambient = ambientStrength * lightColor;

    // Diffuse lighting
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(lightPos - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;

    // Specular lighting
    float specularStrength = 0.5;
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
    vec3 specular = specularStrength * spec * lightColor;

    vec3 result = (ambient + diffuse + specular) * objectColor;
    FragColor = vec4(result, 1.0);
}
"""


def setup_vbo():
    vbo = glGenBuffers(1)
    vao = glGenVertexArrays(1)

    glBindVertexArray(vao)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_DYNAMIC_DRAW)

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * vertices.itemsize, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)

    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glBindVertexArray(0)

    return vao, vbo


def update_vbo(vbo, verticess):
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferSubData(GL_ARRAY_BUFFER, 0, verticess.nbytes, verticess)
    glBindBuffer(GL_ARRAY_BUFFER, 0)


# def draw_interpolated_object(t, vao, vbo, cube_vertices, sphere_v):
#     interpolated = interpolate_vertices(t, cube_vertices, sphere_v)
#     #update_vbo(vbo, interpolated)
#
#     glBindVertexArray(vao)
#     glDrawArrays(GL_TRIANGLES, 0, len(interpolated))
#     glBindVertexArray(0)


# Компиляция шейдера
def compile_shader(shader_type, source):
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)
    if glGetShaderiv(shader, GL_COMPILE_STATUS) != GL_TRUE:
        raise RuntimeError(glGetShaderInfoLog(shader).decode())
    return shader


# Программа шейдера
def create_shader_program(vertex_src, fragment_src):
    vertex_shader = compile_shader(GL_VERTEX_SHADER, vertex_src)
    fragment_shader = compile_shader(GL_FRAGMENT_SHADER, fragment_src)
    shader_program = glCreateProgram()
    glAttachShader(shader_program, vertex_shader)
    glAttachShader(shader_program, fragment_shader)
    glLinkProgram(shader_program)
    if glGetProgramiv(shader_program, GL_LINK_STATUS) != GL_TRUE:
        raise RuntimeError(glGetProgramInfoLog(shader_program).decode())
    glDeleteShader(vertex_shader)
    glDeleteShader(fragment_shader)
    return shader_program


def setLight():
    position = [0, 0, 1, 1]
    if setInfinityDistantLight:
        position[3] = 0
    else:
        position[3] = 1

    glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, [0.3, 0.3, 0.3, 1])
    glLightfv(GL_LIGHT0, GL_POSITION, position)
    glLightfv(GL_LIGHT0, GL_AMBIENT, ambient[ambientMode])
    glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuse[diffuseMode])
    glLightfv(GL_LIGHT0, GL_SPECULAR, specular[specularMode])


def projection(f, a):
    return np.array([
        [np.cos(f), np.sin(f) * np.sin(a), -np.sin(f) * np.cos(a), 0.0],
        [0.0, np.cos(a), np.sin(a), 0.0],
        [np.sin(f), -np.sin(a) * np.cos(f), np.cos(a) * np.cos(f), 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])


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




def subdivide_face(total_expected_points):
    """
    Разбивает грань, заданную четырьмя вершинами, на равномерную сетку.
    :param v1, v2, v3, v4: вершины грани
    :param divisions: количество делений вдоль одной стороны
    :return: список точек на грани
    """
    divisions = 0
    count_point_in_face = (2 * (divisions + 1) ** 2 - 2 * (divisions + 1))
    while count_point_in_face * 6 < total_expected_points:
        divisions += 1
        count_point_in_face = (2 * (divisions + 1) ** 2 - 2 * (divisions + 1))

    points = []
    for k, face in enumerate(faces):  # Для каждой грани куба
        v1 = vertices[face[0]]
        v2 = vertices[face[1]]
        v3 = vertices[face[2]]
        v4 = vertices[face[3]]

        for i in range(divisions):
            for j in range(divisions + 1):
                # Верхняя точка текущей полосы
                p1 = v1 + (v2 - v1) * (j / divisions)
                p2 = v4 + (v3 - v4) * (j / divisions)
                point1 = p1 + (p2 - p1) * (i / divisions)
                point2 = p1 + (p2 - p1) * ((i + 1) / divisions)

                # Добавляем поочерёдно точки двух полос
                points.append(point1)
                points.append(point2)

    # Если точек больше, чем нужно
    if len(points) > total_expected_points:
        # Избыточное количество точек
        extra_points = len(points) - total_expected_points
        face_row_points = (divisions + 1) * 2
        faces_to_process = 1

        # Определяем, сколько точек нужно удалить для каждой грани
        while face_row_points * faces_to_process * 6 < extra_points:
            faces_to_process += 1

        points_to_remove = face_row_points * faces_to_process * 6
        removed_points = points_to_remove

        # print(face_row_points, points_to_remove, faces_to_process)

        # Состояния для управления удалением
        last_face_index = 5
        first_strip = True
        removed_count = 0
        start_index = 0
        end_index = 0
        remove_alternate = False  # Управляет удалением через одну или подряд. True - через одну. False - подряд

        for i in range(len(points) - 1, -1, -1):  # Итерация с конца
            current_face_index = i // count_point_in_face

            # Пропуск следующих точек
            if remove_alternate:
                remove_alternate = False
                continue

            if count_point_in_face - i % count_point_in_face > 1:
                if first_strip:
                    first_strip = False
                    start_index = i

                # Удаляем точки
                if points_to_remove - removed_points < face_row_points * faces_to_process:
                    points.pop(i)
                    removed_points -= 1
                    end_index = i
                    removed_count += 1
                    # print(points_to_remove, removed_points, face_row_points, faces_to_process)

                    remove_alternate = True
                    # Управляем состоянием пропуска
                    # if removed_count % (divisions+1) == 0:
                    #     if removed_count != (divisions + 1) * 2 * faces_to_process - (divisions + 1):
                    #         remove_alternate = False
                    #     else:
                    #         remove_alternate = True
                    # if faces_to_process > 1:
                    # print(removed_count)
                    if (divisions + 1) <= removed_count <= (divisions + 1) * 2 * faces_to_process - (divisions + 1):
                        remove_alternate = False

            # Переход к следующей грани
            if last_face_index != current_face_index or i < 1:
                face_row_points += (divisions + 1) * 2
                last_face_index = current_face_index
                removed_count = 0
                first_strip = True
                remove_alternate = False  # Возвращаемся к удалению через одну

                # Шафлим оставшиеся точки для корректного порядка
                points = custom_shuffle(points, end_index - 1, start_index - (divisions + 1) * 2 * faces_to_process)


    missing_points = total_expected_points - len(points)
    # Делим массив на 6 частей
    num_faces = 6
    face_size = len(points) // num_faces  # Предполагаем, что массив точек можно разделить на равные части

    for face_index in range(num_faces - 1, -1, -1):
        start = face_index * face_size
        end = (face_index + 1) * face_size if face_index < num_faces - 1 else len(points)
        # Вычисляем, сколько точек добавить в текущую грань
        points_to_add = missing_points // num_faces
        if face_index < missing_points % num_faces:
            points_to_add += 1  # Распределяем остаток равномерно

        i = end - 3
        while points_to_add >= 3:
            point = points[i:i + 3]
            for j in range(2, -1, -1):
                points.insert(i, point[j])
            points_to_add -= 3
            i -= 3

        while points_to_add != 0:
            point = points[end - 1]
            points.insert(end - 1, point)
            points_to_add -= 1

    print("Final number of points:", len(points))

    return np.array(points, dtype=np.float32)






def draw_cube():
    # Масштабируем вершины
    scaled_vertices = vertices * (1 + size)

    glBegin(GL_QUADS)
    for face_index, face in enumerate(faces):
        glColor3f(*colors[face_index])  # Установить цвет для грани
        for vertex in face:
            glVertex3f(*scaled_vertices[vertex])  # Используем масштабированные вершины
    glEnd()


def display(window):
    global t, direction
    # Clear the color and depth buffers
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    #setLight()

    # Set the viewport
    width, height = glfw.get_framebuffer_size(window)
    glViewport(0, 0, width, height)

    # Set the projection matrix
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glMultMatrixf(projection(np.radians(45), -np.radians(35)))

    # Set the modelview matrix
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    glPushMatrix()
    v = np.array([1.0, 0.0, 1.0])
    glMultMatrixf(rotation_matrix(v, np.radians(angle)))
    v = np.array([0.0, 1.0, 0.0])
    glMultMatrixf(rotation_matrix(v, np.radians(angle_2)))

    if morph:
        t += 0.001 * direction
        if t > 1.0 or t < 0.0:
            direction *= -1
    draw_interpolated_object(t)  # Draw the cube
    # arr = generate_divided_cube(4)
    # glBegin(GL_TRIANGLE_STRIP)
    # for point in arr:
    #     glVertex3f(*point)
    # glEnd()

    glPopMatrix()

    if switch:  # switching between wireframe and solid-state model display
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
    else:
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

    # Swap the front and back buffers
    glfw.swap_buffers(window)

    # Poll for and process events
    glfw.poll_events()


def interpolate_vertices(t, v1, v2):
    return (1 - t) * v1 + t * v2


def draw_interpolated_object(t):
    global vertices
    sphere_v = sphere.generate_sphere_1()

    if len(sphere_v) > 8:
        vertices_cube = subdivide_face(len(sphere_v))
    else:
        raise ValueError("Куб должен иметь более 8 вершин.")

    interpolated = interpolate_vertices(t, vertices_cube, sphere_v)
    normals = morphing.update_normals(interpolated)

    glBegin(GL_TRIANGLE_STRIP)
    for vertex in interpolated:
        glColor3f(abs(vertex[0]), abs(vertex[1]), abs(vertex[2]))
        glVertex3f(vertex[0], vertex[1], vertex[2])
    glEnd()

    draw_normals(interpolated, normals)


# Функция для рисования нормалей
def draw_normals(vertices, normals, length=0.1):
    glBegin(GL_LINES)
    for i in range(len(vertices)):
        # Точка, из которой начинается нормаль
        x, y, z = vertices[i]
        # Направление нормали
        nx, ny, nz = normals[i]

        # Отображение линии от вершины в сторону нормали (с учетом длины)
        glColor3f(1.0, 0.0, 0.0)  # Красный цвет для нормали
        glVertex3f(x, y, z)  # Начало линии (вершина)
        glVertex3f(x + nx * length, y + ny * length, z + nz * length)  # Конец линии (вершина + нормаль)
    glEnd()

def main():
    # Initialize the GLFW library
    if not glfw.init():
        return
    # Create the window
    window = glfw.create_window(640 * 2, 2 * 640, "Lab2", None, None)
    if not window:
        glfw.terminate()
        return

    # Make the window's context current
    glfw.make_context_current(window)
    glfw.set_key_callback(window, key_callback)
    glfw.set_scroll_callback(window, scroll_callback)
    # Компиляция шейдеров и создание программы
    shader_program = create_shader_program(VERTEX_SHADER_SOURCE, FRAGMENT_SHADER_SOURCE)


    while not glfw.window_should_close(window):
        display(window)


    # Terminate GLFW
    glfw.terminate()


def key_callback(window, key, scancode, action, mods):
    global angle, angle_2, switch, size, t, morph
    if action == glfw.PRESS:
        if key == glfw.KEY_RIGHT:
            angle_2 += 10
        if key == 263:  # glfw.KEY_LEFT
            angle_2 -= 10
        if key == glfw.KEY_UP:
            angle -= 10
        if key == glfw.KEY_DOWN:
            angle += 10
        if key == glfw.KEY_SPACE:
            angle = 0.0
            angle_2 = 0.0
            size = 0.0
        if key == glfw.KEY_S:
            switch = not switch
        if key == glfw.KEY_1:
            t = 0
        if key == glfw.KEY_2:
            t = 1
        if key == glfw.KEY_Q:
            morph = not morph


def scroll_callback(window, xoffset, yoffset):
    global size
    if (xoffset > 0):
        size -= yoffset / 10
    else:
        size += yoffset / 10


if __name__ == '__main__':
    main()
