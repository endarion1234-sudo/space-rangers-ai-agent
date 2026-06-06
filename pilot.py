from system_utils import SystemHandler
import numpy as np



class Pilot:
    def __init__(self, game_region, window_title, cfg):
        self.game_region = game_region
        self.window_title = window_title
        self.cfg = cfg
        self.hwnd = None
        self.camera_position = np.array([0.0, 0.0])
    
    def get_cursor_position(self):
        """
        Получает позицию курсора через системный обработчик.
        """
        # Просто вызываем статический метод из нашего нового класса
        return SystemHandler.get_cursor_position()

    
    def move_cursor_in_game(self, action):
        """
        Перемещает курсор на величину 'action'.
        'action' - это уже масштабированный вектор смещения.
        """
        current_x, current_y = SystemHandler.get_cursor_position()
        # Просто применяем действие как есть!
        new_x = int(current_x + action[0])
        new_y = int(current_y + action[1])

        SystemHandler.set_cursor_position(new_x, new_y)
    
    def set_camera(self, position):
        """Метод для обновления позиции камеры из игры."""
        self.camera_position = np.array(position)
    
    def screen_to_world(self, screen_pos):
        """
        Преобразует координаты из системы экрана в систему игрового мира.
        Это аналог вашей функции ScreenToWorld.
        """
        # Приводим к numpy-массивам для удобства вычислений
        screen_pos = np.array(screen_pos)
        return screen_pos - self.camera_position
    
        

       