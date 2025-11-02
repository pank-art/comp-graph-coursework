import glfw
from OpenGL.GL import *


def init_window(width=800, height=600, title="OpenGL Window"):
    if not glfw.init():
        raise Exception("GLFW could not be initialized")

    window = glfw.create_window(width, height, title, None, None)
    if not window:
        glfw.terminate()
        raise Exception("GLFW window could not be created")


    # Установка контекста OpenGL для окна
    glfw.make_context_current(window)

    glViewport(0, 0, width, height)
    glEnable(GL_DEPTH_TEST)

    glClearColor(0.2, 0.3, 0.3, 1.0)

    return window


def terminate():
    glfw.terminate()
