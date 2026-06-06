import pyautogui
import time
import logging

# Настраиваем логгер для этого модуля
logger = logging.getLogger(__name__)

class ButtonManager:
    """
    Класс для управления игрой через горячие клавиши.
    Реализует дискретное управление.
    """

    def __init__(self, move_duration=0.1):
        """
        :param move_duration: Длительность нажатия клавиш движения (W, A, S, D) в секундах.
        """
        self.move_duration = move_duration
        pyautogui.FAILSAFE = False # Отключаем защиту, если управление стабильное
        logger.debug(f"[BUTTON] Инициализация завершена. Длительность шага: {move_duration}с")

    # --- ОСНОВНЫЕ МЕТОДЫ ВВОДА ---

    def press_key(self, key, duration=None):
        """Нажимает и удерживает клавишу заданное время."""
        if duration is None:
            duration = self.move_duration
            
        logger.debug(f"[BUTTON] Нажимаю: {key} на {duration} сек.")
        pyautogui.keyDown(key)
        time.sleep(duration)
        pyautogui.keyUp(key)

    def press_once(self, key):
        """Нажимает и сразу отпускает клавишу."""
        logger.debug(f"[BUTTON] Клик: {key}")
        pyautogui.press(key)

    # --- ДВИЖЕНИЕ (Дискретное управление) ---

    def weapon(self):
        """Открывает панель вооружения"""
        self.press_key('w')

    def move_down(self):
        """Открывает панель корабля"""
        self.press_key('s')
    
    def menu(self):
        """меню"""
        self.press_key('Esc')

    def jurnal(self):
        """бортовой журнал"""
        self.press_key('F1')
    
    def save(self):
        """Меню сохранения"""
        self.press_key('F2')
    def load(self):
        """Меню загрузки"""
        self.press_key('F3')
    
    def quick_save(self):
        """quick_save"""
        self.press_key('F5')
    
    def load_quick_save3(self):
        """згрузить quick_save3"""
        self.press_key('F6')
    
    def load_quick_save2(self):
        """згрузить quick_save2"""
        self.press_key('F7')
    
    def load_quick_save1(self):
        """згрузить quick_save1"""
        self.press_key('F8')
    
    def all_weapon(self):
        """выбрать всё оружие"""
        self.press_key('`')
    
    def weapon1(self):
        """взять оружие слота 1"""
        self.press_key('1')
    
    def weapon2(self):
        """взять оружие слота 2"""
        self.press_key('2')
    
    def weapon3(self):
        """взять оружие слота 3"""
        self.press_key('3')
    
    def weapon4(self):
        """взять оружие слота 4"""
        self.press_key('4')
    
    def weapon5(self):
        """взять оружие слота 5"""
        self.press_key('5')
    
    def rank_board(self):
        """открывает рейтинговую таблицу"""
        self.press_key('r')
    
    def task(self):
        """Разговор с кораблями"""
        self.press_key('t')
    
    def scan_ship(self):
        """сканирование кораблей"""
        self.press_key('i')

    def centr_camera(self):
        """центрирует камеру"""
        self.press_key('c')
    
    def global_map(self):
        """открывает глобальную карту"""
        self.press_key('m')
    
    def move_turn(self):
        """Запуск хода"""
        self.press_key('Space')
    
    def camera_up(self):
        """перемещает камеру вверх"""
        self.press_key('up')
    
    def camera_down(self):
        """перемещает камеру вниз"""
        self.press_key('down')
    
    def camera_left(self):
        """перемещает камеру влево"""
        self.press_key('left')
    
    def camera_right(self):
        """перемещает камеру вправо"""
        self.press_key('right')

    
    

    # (Кнопки N, L, Esc и т.д. можно добавить сюда по аналогии)