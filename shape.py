import numpy as np
from OpenGL.GL import *


class Shape:
    def __init__(self):
        self.vertices = None

    def draw(self):
        glBegin(GL_TRIANGLE_STRIP)
        for vertex in self.vertices:
            glColor3f(abs(vertex[0]), abs(vertex[1]), abs(vertex[2]))
            glVertex3f(vertex[0], vertex[1], vertex[2])
        glEnd()


class SPHERE(Shape):
    def __init__(self, lat_segments=5, lon_segments=9, radius=0.5):
        super().__init__()
        self.lat_segments = lat_segments
        self.lon_segments = lon_segments
        self.radius = radius

        self.generate_sphere()

    def generate_sphere(self):
        vertices = []
        for i in range(self.lat_segments):
            for j in range(self.lon_segments + 1):  # Добавляем +1 для замыкания по долготе
                # Точка текущей полосы
                phi1 = np.pi * i / self.lat_segments
                theta = 2 * np.pi * j / self.lon_segments
                x1 = self.radius * np.sin(phi1) * np.cos(theta)
                y1 = self.radius * np.sin(phi1) * np.sin(theta)
                z1 = self.radius * np.cos(phi1)

                # Точка следующей полосы
                phi2 = np.pi * (i + 1) / self.lat_segments
                x2 = self.radius * np.sin(phi2) * np.cos(theta)
                y2 = self.radius * np.sin(phi2) * np.sin(theta)
                z2 = self.radius * np.cos(phi2)

                # Добавляем вершины поочередно
                vertices.append((x1, y1, z1))
                vertices.append((x2, y2, z2))

        self.vertices = np.asarray(vertices, dtype=np.float32)


class CUBE(Shape):
    def __init__(self, total_expected_points):
        # Координаты вершин куба (центрированного в (0,0,0) с длиной ребра 0.6)
        super().__init__()
        self.vertices = np.array([
            [-0.3, -0.3, 0.3],  # Вершина 0
            [0.3, -0.3, 0.3],  # Вершина 1
            [0.3, 0.3, 0.3],  # Вершина 2
            [-0.3, 0.3, 0.3],  # Вершина 3
            [-0.3, -0.3, -0.3],  # Вершина 4
            [0.3, -0.3, -0.3],  # Вершина 5
            [0.3, 0.3, -0.3],  # Вершина 6
            [-0.3, 0.3, -0.3]  # Вершина 7
        ])
        # Грани куба (по индексу вершин)
        self.faces = [
            [1, 2, 3, 0],  # Передняя
            [3, 2, 6, 7],  # Верхняя грань
            [6, 5, 1, 2],  # Правая грань
            [5, 4, 0, 1],  # Нижняя грань
            [0, 4, 7, 3],  # Левая грань
            [7, 4, 5, 6],  # Задняя грань
        ]
        self.normals = None

        self._subdivide_face(total_expected_points)
        self._update_normals()

    def _subdivide_face(self, total_expected_points):
        """
        Разбивает все грани куба с 8 вершинами на равномерные сетку.
        :param total_expected_points: необходимое количество точек на всем кубе
        :return: список всех точек
        """
        if total_expected_points < 16:
            raise "Нехватка вершин"

        divisions = 0
        count_point_in_face = (2 * (divisions + 1) ** 2 - 2 * (divisions + 1))
        c_p_i_f_2 = count_point_in_face
        while count_point_in_face * 6 < total_expected_points:
            c_p_i_f_2 = count_point_in_face
            divisions += 1
            count_point_in_face = (2 * (divisions + 1) ** 2 - 2 * (divisions + 1))

        points_to_add_1 = count_point_in_face
        while points_to_add_1 * 6 > total_expected_points:
            points_to_add_1 -= (divisions + 1) * 2

        points_to_add_1 = total_expected_points - points_to_add_1 * 6
        points_to_add_2 = total_expected_points - c_p_i_f_2 * 6

        if points_to_add_2 < points_to_add_1:
            count_point_in_face = c_p_i_f_2
            divisions -= 1

        points = []
        for k, face in enumerate(self.faces):  # Для каждой грани куба
            v1 = self.vertices[face[0]]
            v2 = self.vertices[face[1]]
            v3 = self.vertices[face[2]]
            v4 = self.vertices[face[3]]

            for i in range(divisions):
                for j in range(divisions + 1):
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

            last_face_index = 5
            first_strip = True
            removed_count = 0
            start_index = 0
            end_index = 0
            remove_alternate = False  # Управляет удалением через одну или подряд. True - через одну. False - подряд
            for i in range(len(points) - 1, -1, -1):
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

                        remove_alternate = True
                        # Управляем состоянием пропуска
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
                    points = self._custom_shuffle(points, end_index - 1,
                                                  start_index - (divisions + 1) * 2 * faces_to_process)

        missing_points = total_expected_points - len(points)  # Кол-во нехватающих точек
        num_faces = 6
        face_size = len(points) // num_faces  # Массив точек всегда можно разделить на равные 6 частей (грани)
        for face_index in range(num_faces - 1, -1, -1):
            end = (face_index + 1) * face_size if face_index < num_faces - 1 else len(points)
            # Вычисляем, сколько точек добавить в текущую грань
            points_to_add = missing_points // num_faces
            if face_index < missing_points % num_faces:
                points_to_add += 1  # Распределяем остаток равномерно

            i = end - 3
            while points_to_add >= 3:  # Добавляем повторением треугольников
                point = points[i:i + 3]
                for j in range(2, -1, -1):
                    points.insert(i, point[j])
                points_to_add -= 3
                i -= 3
                end += 3

            while points_to_add != 0:  # Добавляем по одной в конец, если нужно
                point = points[end - 1]
                points.insert(end - 1, point)
                points_to_add -= 1

        print("Final number of points:", len(points))

        self.vertices = np.array(points, dtype=np.float32)

    def _custom_shuffle(self, arr, a, b):
        # Получаем подмассив элементов с индексами от a до b (включительно)
        subarray = arr[a:b + 1]

        # Разделяем подмассив на две половины
        mid = (len(subarray) + 1) // 2  # Это будет середина массива
        first_half = subarray[:mid]  # Первая половина
        second_half = subarray[mid:]  # Вторая половина

        # Чередуем элементы из первой и второй половины
        shuffled_subarray = []
        for i in range(len(first_half)):
            shuffled_subarray.append(first_half[i])
            if i < len(second_half):
                shuffled_subarray.append(second_half[i])

        # Создаем новый массив с переставленными элементами
        shuffled_arr = arr[:a] + shuffled_subarray + arr[b + 1:]

        return shuffled_arr

    def _update_normals(self):
        self.normals = np.zeros_like(self.vertices)
        n = len(self.vertices)
        num_faces = 6
        missing_points = n % num_faces  # Сколько точек было добавлено в грани в subdivide_face
        points_in_faces = []
        for face_index in range(num_faces):  # Рассчитываем количество точек в каждой грани
            points = n // num_faces
            if face_index < missing_points % num_faces:
                points += 1

            points += points_in_faces[face_index - 1] if face_index != 0 else 0
            points_in_faces.append(points)

        no_start_tr = set()
        for i in points_in_faces:  # При поиске нормалей НЕ использовать эти точки как начало треугольника, так как конец тогда окажется на другой грани
            no_start_tr.add(i - 1)
            no_start_tr.add(i - 2)

        # Проходим по треугольникам
        for i in range(0, len(self.vertices) - 2, 1):
            if i in no_start_tr:
                continue

            i1, i2, i3 = i, i + 1, i + 2

            # Получаем вершины треугольника
            v1 = self.vertices[i1]
            v2 = self.vertices[i2]
            v3 = self.vertices[i3]

            # Вычисляем векторы сторон
            edge1 = v2 - v1
            edge2 = v3 - v1

            # Нормаль треугольника — векторное произведение
            normal = np.cross(edge1, edge2)

            if np.linalg.norm(normal) != 0:  # Тут разные точки
                # Присваиваем нормаль каждой вершине треугольника
                self.normals[i1] += normal
                self.normals[i2] += normal
                self.normals[i3] += normal

                self.normals[i1] = self.normals[i1] / np.linalg.norm(self.normals[i1])
                self.normals[i2] = self.normals[i2] / np.linalg.norm(self.normals[i2])
                self.normals[i3] = self.normals[i3] / np.linalg.norm(self.normals[i3])
            else:  # Тут несколько одинаковых точек
                self.normals[i2] = self.normals[i1]
                self.normals[i3] = self.normals[i1]

        # Нормализуем все нормали
        self.normals = self.normals / np.linalg.norm(self.normals, axis=1)[:, np.newaxis]

        # Ориентируем нормали наружу (проверяем их ориентацию относительно внешнего вектора)
        # Для этого используем точку внутри объекта (например, центр) и смотрим на угол с нормалью
        center = np.mean(self.vertices, axis=0)  # Центральная точка объекта

        for i in range(len(self.vertices)):
            # Направление от центра к вершине
            direction_to_center = self.vertices[i] - center
            # Скалярное произведение с нормалью
            if np.dot(self.normals[i], direction_to_center) < 0:
                # Если нормаль направлена внутрь (отрицательное скалярное произведение), инвертируем её
                self.normals[i] = -self.normals[i]
