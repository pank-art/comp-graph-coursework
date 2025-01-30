import glfw


def key_callback(window, key, scancode, action, mods, state):
    """
    Обработка нажатий клавиш.
    :param window: окно GLFW
    :param key: клавиша
    :param scancode: аппаратный код клавиши
    :param action: действие (нажата, отпущена, удерживается)
    :param mods: модификаторы (Shift, Ctrl)
    :param state: объект состояния программы
    """
    if action == glfw.PRESS:
        if key == glfw.KEY_SPACE:
            state['rotation_angle_x'] = 0.0
            state['rotation_angle_y'] = 0.0
            state['size'] = 1.2
            state['direction'] = 1.5
        elif key == glfw.KEY_T:
            state['tex_enable'] = not state['tex_enable']
        elif key == glfw.KEY_M:
            state['morph'] = not state['morph']
        elif key == glfw.KEY_F:
            state['fill'] = not state['fill']
        elif key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)
        elif key == glfw.KEY_EQUAL:
            if state['direction'] > 0:
                state['direction'] += 0.2
            else:
                state['direction'] -= 0.2
        elif key == glfw.KEY_MINUS:
            if state['direction'] > 0:
                state['direction'] -= 0.2
            else:
                state['direction'] += 0.2
        elif key == glfw.KEY_UP:
            state['size'] += 0.1
        elif key == glfw.KEY_DOWN:
            state['size'] -= 0.1


def mouse_button_callback(window, button, action, mods, state):
    if button == glfw.MOUSE_BUTTON_LEFT:
        if action == glfw.PRESS:
            state['mouse_pressed'] = True
        elif action == glfw.RELEASE:
            state['mouse_pressed'] = False


def cursor_position_callback(window, xpos, ypos, state):
    """
    Обработка движения курсора.
    :param window: окно GLFW
    :param xpos: координата X курсора
    :param ypos: координата Y курсора
    :param state: объект состояния программы
    """
    if state.get('mouse_pressed', False):
        dx = xpos - state['last_mouse_pos'][0]
        dy = ypos - state['last_mouse_pos'][1]
        state['rotation_angle_x'] += dy * 0.2  # Скорость вращения
        state['rotation_angle_y'] -= dx * 0.2
    state['last_mouse_pos'] = [xpos, ypos]


def scroll_callback(window, xoffset, yoffset, state):
    """
    Обработка прокрутки.
    :param window: окно GLFW
    :param xoffset: смещение по горизонтали
    :param yoffset: смещение по вертикали
    :param state: объект состояния программы
    """
    if xoffset > 0:
        state['size'] -= yoffset / 10
    else:
        state['size'] += yoffset / 10
