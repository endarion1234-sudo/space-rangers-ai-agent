# файл: navigator.py

import logging

logger = logging.getLogger(__name__)

class Navigator:
    """
    Агент-аналитик. Находит объекты на экране.
    Принимает готовый VisionSystem и Pilot.
    """
    def __init__(self, vision_system, pilot):
        """
        :param vision_system: Готовый экземпляр VisionSystem (YOLO).
        :param pilot: Пилот для получения скриншотов.
        """
        logger.debug("[NAVIGATOR] __init__: Начало конструктора.")
        logger.debug(f"[NAVIGATOR] Получен vision_system: {vision_system is not None}")
        logger.debug(f"[NAVIGATOR] Получен pilot: {pilot is not None}")

        self.vision_system = vision_system
        logger.debug("[NAVIGATOR] self.vision_system установлен.")
        self.pilot = pilot
        logger.debug("[NAVIGATOR] Инициализация завершена.")
        logger.debug("[NAVIGATOR] __init__: Конец конструктора. УСПЕХ.")

    def look(self):
        """
        Смотрит на мир и находит цели.
        ИСПРАВЛЕНИЕ: Теперь просто вызывает метод VisionSystem.
        """
        # Нам больше не нужен Pilot здесь! VisionSystem сам возьмет скриншот.
        
        # Просто просим VisionSystem проанализировать текущий мир
        frame = self.pilot.get_screen()
        
        if frame is not None:
            return self.vision_system.analyze_frame(frame)
            
        return None
            
    def get_current_target_coords(self):
        """
        Вспомогательный метод для Капитана.
        Возвращает координаты первой найденной цели.
        """
        game_state = self.look()
        if not game_state: # Если список пустой
            return None
            
        # Возвращаем координаты первого объекта в списке
        return game_state[0]['coords']
    
    def scan_environment(self):
        """
        Запрашивает у VisionSystem данные и возвращает их "как есть".
        Это главная задача Навигатора: быть мостом между зрением и логикой.
        """
        # 1. Запрашиваем список объектов у Зрения
        observations = self.vision_system.get_observations()
        
        # 2. Возвращаем список Капитану (или любому другому компоненту)
        # Здесь НЕТ логики выбора цели!
        # Мы просто передаем всё, что увидели.
        return observations