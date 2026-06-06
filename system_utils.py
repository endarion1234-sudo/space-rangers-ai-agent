# Файл: system_utils.py

import ctypes
import logging

# Настраиваем логгер для этого модуля
logger = logging.getLogger(__name__)

class SystemHandler:
    """
    Класс для взаимодействия с системой через ctypes.
    """
    @staticmethod
    def get_cursor_position():
        """
        Получает текущую позицию курсора, используя ctypes.
        Возвращает кортеж (x, y).
        """
        
            # Определяем структуру POINT, как того требует Windows API
        class POINT(ctypes.Structure):
                _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]
            
        point = POINT()
            # Вызываем функцию из user32.dll
            # Аргумент True (или 1) говорит функции "записать результат в переменную point"
        ctypes.windll.user32.GetCursorPos(ctypes.byref(point))
            
        return (int(point.x), int(point.y))

    def set_cursor_position(x, y):
        """
        Устанавливает позицию курсора в координаты (x, y).
        Использует системную функцию SetCursorPos.
        """
        ctypes.windll.user32.SetCursorPos(int(x), int(y))
            

      