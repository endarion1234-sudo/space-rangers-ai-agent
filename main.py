import yaml
import logging
import time
from pathlib import Path
import torch
from environment import SpaceRangersEnv
from agent import Agent
from training_loop import run_training
from pilot import Pilot
from utils import WindowWatcher
from vision import VisionSystem
from navigator import Navigator
from captain import Captain
from datetime import datetime as dt
from screen_ai import ScreenProcessor
import sys
import numpy as np
from extractor import CustomVecExtractor
import warnings


warnings.filterwarnings('ignore', module='comtypes.*')
# --- НАСТРОЙКА ЛОГГЕРА (в самом начале файла) ---
logger = logging.getLogger()
logger.setLevel(logging.DEBUG) # Включаем самый подробный уровень

# 1. Обработчик для КОНСОЛИ (stdout)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO) # В консоль выводим только INFO и выше, чтобы не было спама
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

# 2. Обработчик для ФАЙЛА (debug.log)
# Здесь оставляем DEBUG, чтобы в файл записывалось всё.
file_handler = logging.FileHandler('debug.log', mode='w')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

# Добавляем обработчики к корневому логгеру
logger.addHandler(console_handler)
logger.addHandler(file_handler)

predictor_logger = logging.getLogger("ultralytics.engine.predictor")
predictor_logger.setLevel(logging.CRITICAL) 
predictor_logger.propagate = False 

print("[MAIN] Логирование YOLO отключено.")

def load_config():
    """Загружает конфигурацию из файла config.yaml."""
    config_path = Path(__file__).parent / 'config.yaml'
    try:
        with open(config_path, 'r', encoding='utf-8') as stream:
            return yaml.safe_load(stream)
    except FileNotFoundError:
        logger.error(f"Файл конфигурации не найден: {config_path}")
        raise
    except Exception as e:
        logger.error(f"Ошибка при чтении конфига: {e}")
        raise


def main():
    """Главная функция, запускающая цикл обучения."""
    logger.info("[START]===== Новая сессия 1 Started at 2026-05-19 08:44:32 =====")
    
    try:
        cfg = load_config()
        logger.info("[MAIN] Загрузка конфигурации...")
        watcher = WindowWatcher(window_title=cfg.get('WINDOW_TITLE', "Rangers"), timeout=30)
        logger.info("[MAIN] Ожидание активации окна игры...")
        watcher.wait_for_active()
        time.sleep(2.0) 
        pilot = Pilot(tuple(cfg['GAME_REGION']), window_title=cfg.get('WINDOW_TITLE', "Rangers"), cfg=cfg)
        logger.info("[DEBUG] [SUCCESS] Объект Pilot создан успешно.")
        screen_processor = ScreenProcessor(game_region=tuple(cfg['GAME_REGION']), cfg=cfg)
        logger.debug("[MAIN] Создан ScreenProcessor.")
        pilot.screen_processor = screen_processor
        model_path = cfg['MODEL_PATH']
        logger.info("[MAIN] Попытка создания VisionSystem...")
        
        vision_system = VisionSystem(
            model_path=model_path,
            pilot_instance=pilot,
            screen_processor=screen_processor
           )

# --- ИСПРАВЛЕННАЯ ПРОВЕРКА ---
        if vision_system is None:
            # Критическая ошибка: объект не создался
            logger.error("[MAIN] Критическая ошибка: VisionSystem не был создан.")
            return 

        # Теперь проверяем флаг заглушки, который мы поставили в __init__
        if hasattr(vision_system, 'is_stub_active') and vision_system.is_stub_active:
            # Это штатная ситуация: объект есть, но модель не загрузилась
            logger.warning("[MAIN] VisionSystem работает в режиме ЗАГЛУШКИ. Модель YOLO не найдена.")
            # Скрипт продолжает работу дальше!
        else:
            # Если мы здесь, значит модель загрузилась успешно
            logger.info("[MAIN] VisionSystem успешно инициализирован с моделью.")
        # 1.3. Создаем Навигатор и Капитана, ПЕРЕДАВАЯ им готовый vision_system
        navigator = Navigator(vision_system=vision_system, pilot=pilot)
        logger.debug("[MAIN] Navigator создан успешно.")

        try:
            logger.info("[MAIN] ГОТОВИМСЯ к созданию SpaceRangersEnv...")
            env = SpaceRangersEnv(
                pilot_instance=pilot, cfg=cfg,
                  screen_processor=screen_processor, vision_system=vision_system)
            
            logger.debug("[MAIN] SpaceRangersEnv создан успешно.")
        except Exception as e:
            logger.error(f"[MAIN] КРИТИЧЕСКАЯ ОШИБКА при создании SpaceRangersEnv: {type(e).__name__}: {e}")
            return
        logger.info("[MAIN] ГОТОВИМСЯ к созданию Agent...")
        policy_kwargs = {
            "net_arch": {
                "pi": [512, 256],
                "vf": [512, 256]
            },
            "activation_fn": torch.nn.LeakyReLU # <-- Обратите внимание: это уже не строка, а сам объект!
        }
        
        try: 
            agent = Agent(
            env, 
            cfg, 
            policy="MultiInputPolicy", 
            policy_kwargs=policy_kwargs,
            )
            obs_space = env.observation_space
            sample = obs_space.sample() # Это то, что библиотека использует для проверки

            logger.error(f"[AGENT] Тип пространства (obs_space): {type(obs_space)}")
            logger.error(f"[AGENT] Пример данных (sample): {type(sample)}")
            if isinstance(sample, dict):
                # Получаем все ключи, чтобы увидеть реальную структуру
                keys = list(sample.keys())
                # Пытаемся получить форму для первого доступного ключа
                # Это безопасно, так как мы проверили, что ключи есть
                first_key = keys[0] 
                sample_shape = str(np.array(sample[first_key]).shape)
                logger.error(f"[AGENT] Ключи в примере: {keys}")
                logger.error(f"[AGENT] Форма первого ключа ('{first_key}'): {sample_shape}")
            else:
                logger.error(f"[AGENT] Тип данных не является словарем. Тип: {type(sample)}") 

            logger.debug("[MAIN] Агент создан успешно.")
            captain = Captain(navigator=navigator, agent=agent)
            env.captain_instance = captain
            logger.debug("[MAIN] Captain создан успешно.")
            
            success = run_training(agent, cfg)
            if success:
                agent.save()
                logger.info("[MAIN] Модель успешно сохранена.")
            
            logger.info("[MAIN] Скрипт завершил работу.")          
        except Exception as e:
            logger.error(f"[MAIN] КРИТИЧЕСКАЯ ОШИБКА в цикле обучения: {type(e).__name__}: {e}")
    except Exception as main_error:
        logging.error(f"[MAIN] Критическая ошибка: {main_error}")
    finally:
        timestamp = dt.now().strftime('%Y-%m-%d %H:%M:%S')
        logger.info(f"===== Сессия завершена в {timestamp} =====")

if __name__ == "__main__":
    main()