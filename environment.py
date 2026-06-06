import gymnasium as gym
import numpy as np
import logging
import random
from collections import deque
import time
import sys
from system_utils import SystemHandler
import torch
import numpy as np
from typing import Optional, Dict, Any, Tuple
from system_utils import SystemHandler


logger = logging.getLogger("ENVIRON")
for handler in logging.getLogger().handlers:
    # Проверяем, является ли он StreamHandler (вывод в консоль)
    if hasattr(handler, 'stream') and handler.stream == sys.stdout:
        handler.flush = sys.stdout.flush
        break


ALL_CLASS_NAMES = [
    'Pirate', 'loot', 'station', 'planet_bh_star', 'dominator',
    'boss', 'ship_coalicia', 'asteroid'
]
NUM_CLASSES = len(ALL_CLASS_NAMES)
CLASS_NAME_TO_INDEX = {name: idx for idx, name in enumerate(ALL_CLASS_NAMES)}

name_space = gym.spaces.Discrete(NUM_CLASSES) 

# 2. Пространство для центра (осталось прежним)
center_space = gym.spaces.Box(low=0, high=1500, shape=(2,), dtype=np.float32)

# 3. Собираем их в Dict (это и есть single_object_space)
single_object_space = gym.spaces.Dict({
    'name': name_space,
    'center': center_space,
})


class SpaceRangersEnv(gym.Env):
    """
    Окружение для реальной игры.
    Связывает все компоненты вместе: Глаза (Vision), Мозг (Agent), Руки (Pilot).
    """
    def __init__(self, pilot_instance, cfg, screen_processor, vision_system):
        # --- ИНИЦИАЛИЗАЦИЯ ПАРАМЕТРОВ ---
        self.vision_system = vision_system
        self.pilot = pilot_instance 
        self.cfg = cfg
        self.screen_processor = screen_processor
        self._previous_observations_list = []
        self._previous_cursor_pos = None 

        # --- НОВАЯ ЛОГИКА: Две точки для сброса ---
        self.corners = [
            np.array([50, 720]),      # Точка А (Левая)
            np.array([2510, 720])     # Точка Б (Правая)
        ]
        self.next_corner_index = 0  # Начинаем с первой точки
        # -----------------------------------------
        # --- ПРОСТРАНСТВО НАБЛЮДЕНИЙ (ИСПРАВЛЕННОЕ) ---
        # 1. Определяем МАКСИМАЛЬНОЕ количество объектов
        MAX_OBJECTS = 100 
        # Каждый объект: [class_index, x, y] -> 3 числа
        VECTOR_DIM = MAX_OBJECTS * 3 

        # Пространство для вектора (наша "визия")
        objects_box_space = gym.spaces.Box(
            low=0, 
            high=1500, 
            shape=(VECTOR_DIM,), 
            dtype=np.float32
        )

        # Пространство для изображения (кадр)
        img_height, img_width = cfg['IMG_HEIGHT'], cfg['IMG_WIDTH']
        image_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(img_height, img_width, 3), # H, W, C (BGR)
            dtype=np.uint8
        )

        # Финальное пространство — это Dict!
        self.observation_space = gym.spaces.Dict({
            'vec_features': objects_box_space,
            'cnn_features': image_space
        })
        
        # --- ПРОСТРАНСТВО ДЕЙСТВИЙ ---
        low = np.array([-100.0, 0, 0.0], dtype=np.float32)
        high = np.array([ 100.0,  0, 1.0], dtype=np.float32)
        self.action_space = gym.spaces.Box(low=low, high=high, shape=(3,), dtype=np.float32)
        # --- ИНИЦИАЛИЗАЦИЯ ПЕРЕМЕННЫХ ---
        try:
            game_region = cfg['GAME_REGION']
            
            # Вычисляем центр для начальной позиции курсора
            try:
                center_x = game_region[0] + game_region[2] // 2
                center_y = game_region[1] + game_region[3] // 2
                initial_center = (center_x, center_y)
            except (TypeError, IndexError):
                print("[ENV] ОШИБКА: Неверный GAME_REGION в конфиге. Используем (0,0).")
                initial_center = (0, 0)

            # Инициализируем очередь для истории кадров
            self.observation_history = deque(maxlen=4)
            
            # Переменные для логики наград и сброса
            self.cursor_pos = np.array(initial_center, dtype=np.float32)
            self.target_pos = None  # Цель будет сгенерирована при сбросе
            self.done = False
            self.step_count = 0

            print("[ENV] Окружение успешно инициализировано.")

        except Exception as e:
            # Если ЛЮБАЯ ошибка произойдет внутри этого блока,
            # мы логируем ее, но НЕ даем ей упасть.
            print(f"[ENV] КРИТИЧЕСКАЯ ОШИБКА в __init__: {e}")
            # Создаем дефолтные значения, чтобы не было None
            self.observation_history = deque(maxlen=4)
            self.cursor_pos = np.array([0, 0], dtype=np.float32)
            self.target_pos = None
            self.done = False
            self.step_count = 0

    def _generate_new_target(self, observations_list):
        """
        Генерирует новую цель (target_pos), ПРАВИЛЬНО переводя координаты
        из пространства кадра YOLO (1280x736) в пространство игры (2560x1440).
        Коэффициенты вычислены как Отношение Игры к Отношению Кадра.
        """
        if not observations_list:
            print("[DEBUG] [ENV] Не удалось сгенерировать цель: список объектов пуст.")
            self.target_pos = None
            return

        # --- ИСПРАВЛЕННЫЕ КОЭФФИЦИЕНТЫ ---
        frame_w, frame_h = 1280, 736 # Размер входного изображения для YOLO
        game_w, game_h = 2560, 1440 # Разрешение вашего игрового окна

        scale_x = game_w / frame_w # 2560 / 1280 = 2.0
        scale_y = game_h / frame_h # 1440 / 736 ≈ 1.956
        
        for obj in observations_list:
            if obj.get('name') == 'station':
                center_x_yolo, center_y_yolo = obj.get('center', (0, 0))
                
                # Применяем правильные коэффициенты
                target_x_game = int(center_x_yolo * scale_x)
                target_y_game = int(center_y_yolo * scale_y)

                self.target_pos = np.array([target_x_game, target_y_game])
                print(f"[DEBUG] [ENV] НОВАЯ ЦЕЛЬ установлена по объекту 'station': {self.target_pos}")
                return

        print("[DEBUG] [ENV] Не удалось сгенерировать цель: объект 'station' не найден.")
        self.target_pos = None

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Сбрасывает окружение в начальное состояние.
        Генерирует цель и формирует первое наблюдение.
        """
        # 1. Сбрасываем внутренние счетчики и флаги
        self.step_count = 0
        self.done = False
        self._previous_observations_list = []

        # 2. Получаем данные с первого кадра
        observations_raw = self.vision_system.get_observations()
        observations_list = observations_raw if isinstance(observations_raw, list) else []

        # 3. Генерируем цель, если объекты найдены
        if observations_list:
            print("[DEBUG] [RESET] Вызываем _generate_new_target с полученными данными.")
            self._generate_new_target(observations_list)
        else:
            print("[DEBUG] [RESET] Объекты не найдены. Цель не сгенерирована.")

        # 4. ФОРМИРУЕМ ПЕРВОЕ НАБЛЮДЕНИЕ (Observation)
        MAX_OBJECTS = 100
        VECTOR_DIM = MAX_OBJECTS * 3
        flat_vector = np.zeros(VECTOR_DIM, dtype=np.float32)

        frame_w, frame_h = 1280, 736
        game_x, game_y, game_w, game_h = self.cfg['GAME_REGION']

        scale_x = game_w / frame_w
        scale_y = game_h / frame_h
        
        next_point = self.corners[self.next_corner_index]
        
        # Перемещаем курсор в эту точку
        print(f"[DEBUG] [RESET] Агент сброшен в точку: {next_point}.")
        SystemHandler.set_cursor_position(int(next_point[0]), int(next_point[1]))

        # Переходим к следующей точке в списке для следующего сброса
        # (0 -> 1 -> 0 -> 1 ...)
        self.next_corner_index = (self.next_corner_index) % len(self.corners)

        # --- ДОБАВЛЯЕМ САМОГО АГЕНТА (КУРСОР) В ПЕРВОЕ НАБЛЮДЕНИЕ ---
        # Это критически важно, чтобы агент знал свою стартовую позицию.
        AGENT = 99
        cursor_x, cursor_y = SystemHandler.get_cursor_position()
        
        flat_vector[0*3 + 0] = AGENT
        flat_vector[0*3 + 1] = cursor_x
        flat_vector[0*3 + 2] = cursor_y
        # -----------------------------------------------------------

        # --- ДОБАВЛЯЕМ ОСТАЛЬНЫЕ ОБЪЕКТЫ ИЗ СПИСКА ---
        for i, obj in enumerate(observations_list, start=1): # Начинаем с индекса 1
            if i >= MAX_OBJECTS:
                break

            class_name = obj.get('name', 'unknown')
            center_x, center_y = obj.get('center', (0, 0))

            scaled_x = int((center_x - game_x) * scale_x)
            scaled_y = int((center_y - game_y) * scale_y)

            class_index = CLASS_NAME_TO_INDEX.get(class_name, 0)

            flat_vector[i*3 + 0] = class_index
            flat_vector[i*3 + 1] = scaled_x
            flat_vector[i*3 + 2] = scaled_y

        frame_for_cnn = self.screen_processor.get_current_frame_for_cv()
        if frame_for_cnn is None:
            img_height, img_width = self.cfg['IMG_HEIGHT'], self.cfg['IMG_WIDTH']
            frame_for_cnn = np.zeros((img_height, img_width, 3), dtype=np.uint8)

        tensor_vector = torch.as_tensor(flat_vector, dtype=torch.float32)
        tensor_frame = torch.as_tensor(frame_for_cnn, dtype=torch.uint8)

        observation = {
            'vec_features': tensor_vector,
            'cnn_features': tensor_frame
        }

        # 5. ПОДГОТОВКА ИНФОРМАЦИИ (Infos)
        infos = {'raw_observations': observations_list}

        return observation, infos
        
    
    def step(self, raw_action):
        """
        Принимает действие от агента, применяет его и возвращает новое состояние,
        награду и информацию о завершении эпизода.
        """        
        # --- 0. ПРОВЕРКА НА ЗАВЕРШЕНИЕ ЭПИЗОДА ---
        if self.done:            
            raise RuntimeError("Episode is done. Call reset().")
   # --- ИНИЦИАЛИЗАЦИЯ ПЕРЕМЕННЫХ ---
        reward = 0.0
        infos = {'raw_observations': []}
        # Выставляем ограничения, иначе ИИ может выдать большие значения
        x_shift_raw = raw_action[0] # Будет в диапазоне [-100, 100]
        y_shift_raw = raw_action[1]     
        # Множитель скорости: приводим из диапазона [-1, 1] в [0.0, 1.0]
        speed_multiplier_raw = raw_action[2]
        # Вычисляем итоговый вектор движения
        print(f"[ARCHI] Число ИИ {raw_action}")
        x_shift = np.clip(x_shift_raw, -100.0, 100.0) # Используем НОВЫЕ границы
        y_shift = 0.0 
        speed_multiplier = (speed_multiplier_raw + 1.0) / 2.0
        move_vector = np.array([x_shift, y_shift]) * speed_multiplier
        reward -= 0.01
        self.pilot.move_cursor_in_game(move_vector)
        
        raw_observations = self.vision_system.get_observations()
        observations_list = raw_observations if isinstance(raw_observations, list) else []
        
        cursor_x, cursor_y = SystemHandler.get_cursor_position()
        current_pos = np.array([cursor_x, cursor_y], dtype=np.float32)


        # --- 3. ЛОГИКА ШТРАФА ЗА ОТСУТСТВИЕ ДВИЖЕНИЯ ---
        if self._previous_cursor_pos is not None:
            movement_delta = np.linalg.norm(current_pos - self._previous_cursor_pos)
            if movement_delta < 1.0:
                reward -= 0.05
                print(f"[DEBUG] [PENALTY] Штраф за отсутствие движения: {movement_delta:.2f} < 1.0")
                
        self._previous_cursor_pos = current_pos


        # --- 4. ФОРМИРОВАНИЕ НАБЛЮДЕНИЯ (Observation) ---
        MAX_OBJECTS = 100
        VECTOR_DIM = MAX_OBJECTS * 3

        frame_w, frame_h = 1280, 736
        game_x, game_y, game_w, game_h = self.cfg['GAME_REGION']

        scale_x = game_w / frame_w
        scale_y = game_h / frame_h

        flat_vector = np.zeros(VECTOR_DIM, dtype=np.float32)

        # ДОБАВЛЯЕМ САМОГО АГЕНТА (КУРСОР) В НАБЛЮДЕНИЕ
        AGENT = 99
        flat_vector[0*3 + 0] = AGENT
        flat_vector[0*3 + 1] = cursor_x
        flat_vector[0*3 + 2] = cursor_y

        # ДОБАВЛЯЕМ ОСТАЛЬНЫЕ ОБЪЕКТЫ ИЗ СПИСКА
        for i, obj in enumerate(observations_list, start=1):
            if i >= MAX_OBJECTS:
                break

            class_name = obj.get('name', 'unknown')
            center_x, center_y = obj.get('center', (0, 0))

            scaled_x = int((center_x - game_x) * scale_x)
            scaled_y = int((center_y - game_y) * scale_y)

            class_index = CLASS_NAME_TO_INDEX.get(class_name, 0)

            flat_vector[i*3 + 0] = class_index
            flat_vector[i*3 + 1] = scaled_x
            flat_vector[i*3 + 2] = scaled_y

        frame_for_cnn = self.screen_processor.get_current_frame_for_cv()
        if frame_for_cnn is None:
            img_height, img_width = self.cfg['IMG_HEIGHT'], self.cfg['IMG_WIDTH']
            frame_for_cnn = np.zeros((img_height, img_width, 3), dtype=np.uint8)

        tensor_vector = torch.as_tensor(flat_vector, dtype=torch.float32)
        tensor_frame = torch.as_tensor(frame_for_cnn, dtype=torch.uint8)

        observation = {
            'vec_features': tensor_vector,
            'cnn_features': tensor_frame
        }


        # --- 5. РАСЧЕТ НАГРАДЫ И ЛОГИКА ПОПАДАНИЯ В ЦЕЛЬ ---
        terminated = False
        truncated = False

        distance_to_target = float('inf')

        if observations_list and self.target_pos is not None:
            target_pos = self.target_pos
            distance_to_target = np.linalg.norm(np.array([cursor_x, cursor_y]) - target_pos)
            HIT_THRESHOLD = 50

            if distance_to_target < HIT_THRESHOLD:
                reward += 20.0
                print(f"[DEBUG] [STEP] ПОПАДАНИЕ! Награда начислена")
                terminated = True


        # --- 6. ОБЩАЯ ЛОГИКА ЗАВЕРШЕНИЯ ЭПИЗОДА (УВЕЛИЧЕН ЛИМИТ) ---
        self.step_count += 1

        # Увеличим лимит шагов, так как с управлением скоростью агенту может требоваться больше времени на маневры.
        if self.step_count > 250: 
            truncated = True


        # --- 7. ВОЗВРАТ РЕЗУЛЬТАТА ---
        infos['raw_observations'] = observations_list
        infos['distance_to_target'] = distance_to_target

        self.done = terminated or truncated
        self._previous_observations_list = observations_list

        return observation, reward, terminated, truncated, infos