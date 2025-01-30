from collections import defaultdict
from OpenGL.GLUT import *
import numpy
import numpy as np
from PIL import Image
import numpy as np
from OpenGL.GL import *


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


def load_shader_code(file_path):
    """
    Загружает код шейдера из файла.
    :param file_path: Путь к файлу шейдера.
    :return: Код шейдера в виде строки.
    """
    with open(file_path, 'r') as file:
        return file.read()


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


def make_texture(filename):
    """
    Загружает текстуру из файла и возвращает идентификатор текстуры OpenGL.

    :param filename: Путь к изображению.
    :return: Идентификатор текстуры OpenGL.
    """
    # Открываем изображение и преобразуем в массив numpy
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

    return texture_id


def generate_texture_coords_sphere(lat_segments=5, lon_segments=9):
    """
    Генерация текстурных координат для сферической поверхности,
    соответствующей порядку вершин из GL_TRIANGLE_STRIP.

    :param lat_segments: Количество полос широты.
    :param lon_segments: Количество сегментов долготы.
    :return: Массив текстурных координат (N x 2).
    """

    tex_coords = []
    for i in range(lat_segments):
        for j in range(lon_segments + 1):  # +1 для замыкания по долготе
            u = j / lon_segments
            v1 = i / lat_segments  # Верхняя широта
            tex_coords.append([u, v1])  # Верхняя точка

            v2 = (i + 1) / lat_segments  # Нижняя широта
            tex_coords.append([u, v2])  # Нижняя точка

    return np.asarray(tex_coords, dtype=np.float32)


def generate_texture_coords_cube(vertices):
    n = len(vertices)
    num_faces = 6
    missing_points = n % num_faces  # Сколько точек было добавлено в грани в subdivide_face
    points_in_faces = []
    for face_index in range(num_faces):  # Рассчитываем количество точек в каждой грани
        points = n // num_faces
        if face_index < missing_points % num_faces:
            points += 1

        points += points_in_faces[face_index - 1] if face_index != 0 else 0
        points_in_faces.append(points)

    points_in_faces.append(0)
    tex_coords = []

    # Генерация текстурных координат для каждой грани
    for i in range(num_faces):
        x = abs(vertices[points_in_faces[i] - 1][0] - vertices[points_in_faces[i - 1]][0])
        y = abs(vertices[points_in_faces[i] - 1][1] - vertices[points_in_faces[i - 1]][1])
        z = abs(vertices[points_in_faces[i] - 1][2] - vertices[points_in_faces[i - 1]][2])
        a = 0
        b = 1

        if x == 0:
            x = z
            a = 2
        if y == 0:
            y = z
            b = 2

        for j in range(points_in_faces[i - 1], points_in_faces[i]):
            u = abs(vertices[j][a] - vertices[points_in_faces[i] - 1][a]) / x
            v = abs(vertices[j][b] - vertices[points_in_faces[i] - 1][b]) / y

            tex_coords.append([u, v])

    return np.asarray(tex_coords, dtype=np.float32)


def setLight(shader_program):
    light_position = [5, 5, 5, 1]

    light_position_location = glGetUniformLocation(shader_program, "lightPosition")
    light_color_location = glGetUniformLocation(shader_program, "lightColor")
    ambient_color_location = glGetUniformLocation(shader_program, "ambientColor")
    diffuse_color_location = glGetUniformLocation(shader_program, "diffuseColor")
    specular_color_location = glGetUniformLocation(shader_program, "specularColor")
    shininess_location = glGetUniformLocation(shader_program, "shininess")

    glUniform3fv(light_position_location, 1, light_position)
    glUniform3fv(light_color_location, 1, [1.0, 0.9, 0.8])
    glUniform3fv(ambient_color_location, 1, [0.5, 0.5, 0.5, 1])
    glUniform3fv(diffuse_color_location, 1, [1, 0.9, 0.8, 1])
    glUniform3fv(specular_color_location, 1, [1, 1, 1, 1])
    glUniform1f(shininess_location, 128.0)


def morph_factor(t, last_time, state):
    EPSILON = 1e-6  # Погрешность для проверки границ
    PAUSE_TIME = 500  # Время задержки на границах в миллисекундах

    time = glutGet(GLUT_ELAPSED_TIME)
    if state.get('morph') == True:
        current_time = time  # Текущее время
        # Проверка: задерживаем t на границах
        if t >= 1.0 - EPSILON:
            t = 1.0
            if current_time - last_time > PAUSE_TIME:
                if state.get('direction') > 0:
                    state['direction'] *= -1  # Меняем направление движения
                last_time = current_time  # Обновляем время последней паузы
                t += 0.001 * state.get('direction')
        elif t <= 0.0 + EPSILON:
            t = 0.0
            if current_time - last_time > PAUSE_TIME:
                if state.get('direction') < 0:
                    state['direction'] *= -1  # Меняем направление движения
                last_time = current_time  # Обновляем время последней паузы
                t += 0.001 * state.get('direction')
        else:
            # t движется непрерывно внутри диапазона (0, 1)
            t += 0.001 * state.get('direction')
            last_time = current_time

    return t, last_time
