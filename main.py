import time
import argparse
import glm
import glfw
from OpenGL.GL.shaders import compileProgram, compileShader
from utils import *
from window_manager import init_window, terminate
from shape import SPHERE, CUBE
from callbacks import key_callback, mouse_button_callback, cursor_position_callback, scroll_callback

WIDTH = 1000
HEIGHT = 800


def display(window, shader_program, vao, state, n, t, last_time):
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glUseProgram(shader_program)
    glBindVertexArray(vao)

    if state.get('tex_enable', True):
        glUniform1i(glGetUniformLocation(shader_program, "tex_enable"), 1)
    else:
        glUniform1i(glGetUniformLocation(shader_program, "tex_enable"), 0)

    v_x = np.array([1.0, 0.0, 0.0])  # Ось X
    v_y = np.array([0.0, 1.0, 0.0])  # Ось Y

    rotation_x = rotation_matrix(v_x, np.radians(state['rotation_angle_x']))
    rotation_y = rotation_matrix(v_y, np.radians(state['rotation_angle_y']))

    model = rotation_x @ rotation_y

    model = model @ np.array([[state.get('size'), 0, 0, 0],
                              [0, state.get('size'), 0, 0],
                              [0, 0, state.get('size'), 0],
                              [0, 0, 0, 1]])

    glUniformMatrix4fv(glGetUniformLocation(shader_program, "model"), 1, GL_FALSE, model)

    morph_factor_location = glGetUniformLocation(shader_program, "morphFactor")
    glUniform1f(morph_factor_location, t)
    glDrawArrays(GL_TRIANGLE_STRIP, 0, n)

    if state.get('fill'):  # switching between wireframe and solid-state model display
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
    else:
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

    # Swap the front and back buffers
    glfw.swap_buffers(window)
    # Poll for and process events
    glfw.poll_events()


def initialize_window_and_state(args):
    window = init_window(WIDTH, HEIGHT, "Morphing")
    state = {
        'rotation_angle_x': 0.0,
        'rotation_angle_y': 0.0,
        'size': 1.2,
        'mouse_pressed': False,
        'last_mouse_pos': (0.0, 0.0),
        'tex_enable': True,
        'morph': False,
        'fill': True,
        'direction': 1.5
    }
    return window, state


def setup_callbacks(window, state):
    # Назначаем обработчики событий
    glfw.set_key_callback(window, lambda w, k, s, a, m: key_callback(w, k, s, a, m, state))
    glfw.set_scroll_callback(window, lambda w, x, y: scroll_callback(w, x, y, state))
    glfw.set_mouse_button_callback(window, lambda w, b, a, m: mouse_button_callback(w, b, a, m, state))
    glfw.set_cursor_pos_callback(window, lambda w, x, y: cursor_position_callback(w, x, y, state))


def load_and_compile_shaders():
    # Загрузка кода шейдеров
    vertex_shader_code = load_shader_code("shaders/vertex_shader.vert")
    fragment_shader_code = load_shader_code("shaders/fragment_shader.frag")

    # Компиляция и связывание шейдеров
    shader_program = compileProgram(
        compileShader(vertex_shader_code, GL_VERTEX_SHADER),
        compileShader(fragment_shader_code, GL_FRAGMENT_SHADER)
    )
    glUseProgram(shader_program)
    return shader_program


def setup_vertex_buffers(sphere, cube, shader_program, lat_segments, lon_segments):
    texture_sphere = generate_texture_coords_sphere(lat_segments, lon_segments)
    texture_cube = generate_texture_coords_cube(cube.vertices)
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)

    vbo_cube = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo_cube)
    glBufferData(GL_ARRAY_BUFFER, cube.vertices.nbytes, cube.vertices, GL_STATIC_DRAW)

    vbo_sphere = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo_sphere)
    glBufferData(GL_ARRAY_BUFFER, sphere.vertices.nbytes, sphere.vertices, GL_STATIC_DRAW)

    vbo_normals = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo_normals)
    glBufferData(GL_ARRAY_BUFFER, cube.normals.nbytes, cube.normals, GL_STATIC_DRAW)

    vbo_texture_cube = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo_texture_cube)
    glBufferData(GL_ARRAY_BUFFER, texture_cube.nbytes, texture_cube, GL_STATIC_DRAW)

    vbo_texture_sphere = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo_texture_sphere)
    glBufferData(GL_ARRAY_BUFFER, texture_sphere.nbytes, texture_sphere, GL_STATIC_DRAW)

    # Получение ссылки на атрибут позиции вершин в шейдере
    position_cube = glGetAttribLocation(shader_program, "cubePosition")
    glBindBuffer(GL_ARRAY_BUFFER, vbo_cube)
    glVertexAttribPointer(position_cube, 3, GL_FLOAT, GL_FALSE, 0, None)
    glEnableVertexAttribArray(position_cube)

    normal_cube = glGetAttribLocation(shader_program, "cubeNormal")
    glBindBuffer(GL_ARRAY_BUFFER, vbo_normals)
    glVertexAttribPointer(normal_cube, 3, GL_FLOAT, GL_FALSE, 0, None)
    glEnableVertexAttribArray(normal_cube)

    position_sphere = glGetAttribLocation(shader_program, "spherePosition")
    glBindBuffer(GL_ARRAY_BUFFER, vbo_sphere)
    glVertexAttribPointer(position_sphere, 3, GL_FLOAT, GL_FALSE, 0, None)
    glEnableVertexAttribArray(position_sphere)

    texture_cube_coord = glGetAttribLocation(shader_program, "textureCube")
    glBindBuffer(GL_ARRAY_BUFFER, vbo_texture_cube)
    glVertexAttribPointer(texture_cube_coord, 2, GL_FLOAT, GL_FALSE, 0, None)
    glEnableVertexAttribArray(texture_cube_coord)

    texture_sphere_coord = glGetAttribLocation(shader_program, "textureSphere")
    glBindBuffer(GL_ARRAY_BUFFER, vbo_texture_sphere)
    glVertexAttribPointer(texture_sphere_coord, 2, GL_FLOAT, GL_FALSE, 0, None)
    glEnableVertexAttribArray(texture_sphere_coord)

    glBindVertexArray(0)

    return vao, vbo_cube, vbo_normals, vbo_sphere


def setup_matrices(shader_program):
    # Установка матрицы проекции
    projection_location = glGetUniformLocation(shader_program, "projection")
    projection = glm.perspective(glm.radians(45.0), WIDTH / HEIGHT, 0.1, 100.0)
    glUniformMatrix4fv(projection_location, 1, GL_FALSE, glm.value_ptr(projection))

    camera_position = np.array([1, 1, -2])  # Камера отодвинута назад
    camera_target = np.array([0, 0, 0])  # Центр фигуры
    up = np.array([0, 1, 0])  # Направление "вверх"
    view = look_at(camera_position, camera_target, up)
    view_location = glGetUniformLocation(shader_program, "view")
    glUniformMatrix4fv(view_location, 1, GL_FALSE, view)

    # Активируем текстурный блок и привязываем текстуру
    texture_id = make_texture('textures/texture.bmp')
    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glUniform1i(glGetUniformLocation(shader_program, "textureSampler"), 0)  # Текстурный блок 0


def main_loop(window, shader_program, vao, state, n, fps):
    last_time = glutGet(GLUT_ELAPSED_TIME)
    t = 0.0
    while not glfw.window_should_close(window):
        start_time = time.time()
        t, last_time = morph_factor(t, last_time, state)
        display(window, shader_program, vao, state, n, t, last_time)
        if (time.time() - start_time) != 0:
            fps.append(1 / (time.time() - start_time))


def cleanup(shader_program, vbo_cube, vbo_normals, vbo_sphere):
    # Освобождение ресурсов
    glDeleteBuffers(1, [vbo_cube])
    glDeleteBuffers(1, [vbo_normals])
    glDeleteBuffers(1, [vbo_sphere])
    glDeleteProgram(shader_program)
    terminate()


def main():
    parser = argparse.ArgumentParser(description='Программа для работы с графикой')
    parser.add_argument('--lat_segments', type=int, default=200, help='Количество сегментов по широте')
    parser.add_argument('--lon_segments', type=int, default=249, help='Количество сегментов по долготе')
    args = parser.parse_args()
    lat_segments, lon_segments = args.lat_segments, args.lon_segments

    window, state = initialize_window_and_state(args)

    setup_callbacks(window, state)

    shader_program = load_and_compile_shaders()

    sphere = SPHERE(lat_segments, lon_segments)
    n = len(sphere.vertices)
    cube = CUBE(n)

    vao, vbo_cube, vbo_normals, vbo_sphere = setup_vertex_buffers(sphere, cube, shader_program, lat_segments,
                                                                  lon_segments)

    setup_matrices(shader_program)

    setLight(shader_program)
    fps = []
    main_loop(window, shader_program, vao, state, n, fps)

    print(f"FPS: {sum(fps) / len(fps)}")
    cleanup(shader_program, vbo_cube, vbo_normals, vbo_sphere)


if __name__ == "__main__":
    main()
