import pyautogui
import time
import logging

# Настраиваем логгер для этого модуля
logger = logging.getLogger(__name__)

class MouseDriver:
    """
    Класс для управления курсором мыши.
    Реализует векторное (плавное) перемещение.
    """
    def __init__(self, move_duration=0.5):
        """
        :param move_duration: Время в секундах, за которое курсор должен долететь до цели.
        """
        self.move_duration = move_duration
        pyautogui.FAILSAFE = False # Отключаем защиту, если управление стабильное
        logger.debug(f"[MOUSE] Инициализация завершена. Скорость движения: {move_duration}с")

    def move_to(self, x, y):
        """
        Плавно перемещает курсор в координаты (x, y).
        """
        logger.debug(f"[MOUSE] Движение к: ({x}, {y})")
        pyautogui.moveTo(x, y, duration=self.move_duration)
        time.sleep(0.01) # Крошечная пауза для стабильности

    def get_position(self):
        """
        Возвращает текущие координаты курсора.
        Используется для проверки достижения цели.
        """
        return pyautogui.position()

    def click(self, button='left'):
        """
        Эмулирует нажатие кнопки мыши.
        """
        pyautogui.click(button=button)
    
    def mouse2(self, button='right'):
        """Эмулирует нажаите правой кнопки мыши """
        pyautogui.click(button=button)

    
