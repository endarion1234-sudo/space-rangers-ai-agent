# файл: captain.py

import logging
import random
import numpy as np

logger = logging.getLogger(__name__)

class Captain:
    """
    Поставщик целей. Получает список объектов и выдает координаты одной цели.
    Не отдает приказы, чтобы не мешать обучению.
    """
    def __init__(self, navigator, agent):
        self.navigator = navigator
        self.current_target_coords = None # Здесь будут храниться координаты цели
        self.agetn=agent

    def receive_intel(self, game_state_objects):
        """
        Получает список объектов от Навигатора и выбирает цель.
        """

        if not game_state_objects:
            self.current_target_coords = None
            return

        # --- НОВАЯ ЛОГИКА: Выбираем цель ---
        # 1. Отбрасываем свой корабль ('self')
        valid_targets = [obj for obj in game_state_objects if obj['type'] != 'self']
        
        if not valid_targets:
            self.current_target_coords = None
            return

        # 2. Выбираем случайную цель из оставшихся
        chosen_target = random.choice(valid_targets)
        
        # 3. Сохраняем координаты цели
        self.current_target_coords = chosen_target['coords']

    
    def get_current_target_coords(self):
        """
        Возвращает координаты текущей цели.
        Этот метод АКТИВНО просит Навигатор найти цель.
        """
        game_state = self.navigator.look()
        
        if game_state: 
            # Если цели есть, берем первую
            return game_state[0]['coords']
        else:
            # Если целей нет, возвращаем None
            return None
    
    def get_target_vector(self):
        """
        Возвращает ТОЛЬКО вектор цели [x_tgt, y_tgt].
        Это "чистый" метод Капитана, который не знает о Пилоте.
        """
        
            # Просим Навигатор найти цель
        game_state = self.navigator.look()
            
        if not game_state: 
                # Если целей нет, возвращаем None (пусть Окружение само ставит цель в центр)
                return None
                
            # Возвращаем координаты первой найденной цели как вектор [x, y]
        tgt_x, tgt_y = game_state[0]['coords']
        return np.array([tgt_x, tgt_y], dtype=np.float32)

      
    
    # В файле captain.py
    def make_decision(self):
        # 1. Получаем СПИСОК всех объектов от Зрения
        observations = self.navigator.scan_environment() 
        
        # 2. Передаем этот СПИСОК напрямую ИИ
        # (ИИ сам решит, кто из них враг, а кто союзник)
        action = self.agent.choose_action(observations)
        
        return action