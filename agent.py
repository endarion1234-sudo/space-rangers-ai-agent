import os
import re
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed

# Настраиваем логгер для этого модуля
logger = logging.getLogger("AGENT")

class Agent:
    """
    Класс-обертка для модели PPO.
    Отвечает за жизненный цикл модели: создание, загрузку, сохранение.
    Не содержит логики обучения (learn), чтобы не нарушать принцип SRP.
    """

    def __init__(self, env, cfg: Dict[str, Any], policy: str = "MlpPolicy", policy_kwargs: Optional[Dict] = None):
        """
        Инициализирует агента.
        :param env: Окружение (Env).
        :param cfg: Конфигурация (словарь).
        :param policy: Тип политики (например, "MlpPolicy").
        :param policy_kwargs: Аргументы для архитектуры сети.
        """
        self.env = env
        self.cfg = cfg
        
        # --- Пути к папкам ---
        self.checkpoints_dir = Path(cfg.get('CHECKPOINTS_DIR', 'checkpoints'))
        self.saved_models_dir = Path(cfg.get('SAVED_MODELS_DIR', 'saved_models'))
        
        # Путь к файлу latest (стандарт для SB3)
        self.model_path_latest = self.checkpoints_dir / "rl_model_latest.zip"
        self.model_path: Optional[Path] = None # Путь к файлу, который будет загружен
        # --- 1. ПОИСК МОДЕЛИ ДЛЯ ЗАГРУЗКИ ---
        logger.info("[AGENT] Поиск существующей модели для загрузки...")
        
        # Приоритет 1: Загрузка из промежуточных чекпоинтов
        if self._find_latest_checkpoint():
            logger.info(f"[AGENT] Найден чекпоинт для загрузки: {self.model_path}")
        # Приоритет 2: Загрузка из финальных сохранений
        elif self._find_latest_saved_model():
            logger.info(f"[AGENT] Найдена сохраненная модель для загрузки: {self.model_path}")
        else:
            logger.info("[AGENT] Существующих моделей не найдено. Подготовка к созданию новой.")
            self._cleanup_checkpoints_dir() # Чистим чекпоинты перед новым обучением
            self._set_random_seed() # Устанавливаем сид для воспроизводимости

        # --- 2. СОЗДАНИЕ ИЛИ ЗАГРУЗКА МОДЕЛИ ---
        if self.model_path and self.model_path.exists():
            self._load_existing_model()
        else:
            self._create_new_model(policy, policy_kwargs)

    def _set_random_seed(self):
        """Устанавливает глобальный сид для воспроизводимости."""
        seed = self.cfg.get('SEED', 0)
        logger.info(f"[AGENT] Установка глобального сида для воспроизводимости: {seed}")
        set_random_seed(seed)

    def _find_latest_checkpoint(self) -> bool:
        """Ищет самый новый файл в папке checkpoints."""
        checkpoint_files = list(self.checkpoints_dir.glob("rl_model_*.zip"))
        if not checkpoint_files:
            return False

        def get_step_from_path(p: Path) -> int:
            match = re.search(r'_(\d+)_steps\.zip$', str(p.name))
            return int(match.group(1)) if match else -1

        latest_file = max(checkpoint_files, key=get_step_from_path)
        
        if latest_file:
            self.model_path = latest_file
            return True
        return False

    def _find_latest_saved_model(self) -> bool:
        """Ищет самый новый файл в папке saved_models."""
        saved_files = list(self.saved_models_dir.glob("*.zip"))
        if not saved_files:
            return False
        
        # Берем файл с самой поздней датой изменения
        latest_saved = max(saved_files, key=os.path.getmtime)
        
        if latest_saved:
            self.model_path = latest_saved
            return True
        return False

    def _cleanup_checkpoints_dir(self):
        """Очищает папку чекпоинтов перед новым обучением.
        """
        logger.info("[AGENT] Очистка папки checkpoints для новой сессии обучения.")
        for file_path in self.checkpoints_dir.glob("*"):
            try:
                if file_path.is_file():
                    file_path.unlink()
                    logger.debug(f"[AGENT] Удален файл: {file_path.name}")
            except Exception as e:
                logger.warning(f"[AGENT] Не удалось удалить файл {file_path.name}: {e}")

    def _get_ppo_kwargs(self, policy_kwargs_from_main: Optional[Dict]) -> Dict[str, Any]:
        """
        Собирает аргументы для PPO.
        Приоритет 1: Настройки, переданные напрямую в Agent (policy_kwargs_from_main).
        Приоритет 2: Настройки из конфига self.cfg.
        """
        # Создаем базовый словарь настроек из конфига
        base_kwargs = {
            "verbose": self.cfg.get('VERBOSE', 1),
            "device": self.cfg.get('DEVICE', "auto"),
            "learning_rate": self.cfg.get('LEARNING_RATE', 3e-4),
            "batch_size": self.cfg.get('BATCH_SIZE', 128),
            "gamma": self.cfg.get('GAMMA', 0.99),
            "clip_range": self.cfg.get('CLIP_RANGE', 0.2),
            "tensorboard_log": self.cfg.get('TENSORBOARD_LOG', "./logs"),
        }
        # --- НОВОЕ РЕШЕНИЕ: ---
        # Если настройки переданы извне, используем их как базу.
        # Иначе используем пустую dict, чтобы не было None.
        if policy_kwargs_from_main is not None:
            final_policy_kwargs = policy_kwargs_from_main
        return {
            **base_kwargs,
            "policy_kwargs": final_policy_kwargs,
        }

    def _create_new_model(self, policy: str, policy_kwargs: Optional[Dict]):
        """Создает новую модель PPO с нуля."""
        logger.info(f"[AGENT] Создание новой модели с политикой: {policy}")
        
        self.model = PPO(
            policy, 
            self.env,
            **self._get_ppo_kwargs(policy_kwargs)
        )
        logger.info("[AGENT] Новая модель успешно создана.")

    def _load_existing_model(self):
        """Загружает существующую модель из файла."""
        logger.info(f"[AGENT] Загрузка существующей модели из файла: {self.model_path}")
        try:
            # Загружаем модель, передавая окружение для восстановления векторов нормализации
            self.model = PPO.load(self.model_path, env=self.env)
            logger.info("[AGENT] Модель успешно загружена.")
            
            # Пытаемся получить шаг, с которого продолжаем обучение
            if hasattr(self.model, "get_vec_normalize_env") and self.model.get_vec_normalize_env():
                current_step = self.model.get_vec_normalize_env().num_timesteps
                logger.info(f"[AGENT] Продолжение обучения с шага: {current_step}")
                
        except Exception as e:
            logger.error(f"[AGENT] ОШИБКА при загрузке модели: {e}")
            raise # Пробрасываем ошибку вверх, чтобы остановить скрипт

    def get_model(self):
        """Возвращает обучаемую модель для запуска обучения."""
        if not hasattr(self, 'model'):
            raise RuntimeError("Модель не была инициализирована.")
        return self.model

    def save_final_model(self, name_suffix: str = "best"):
        """
        Сохраняет финальную/лучшую модель в папку saved_models.
        :param name_suffix: Суффикс для имени файла (например, 'best', 'final_v2').
        """
        filename = f"final_model_{name_suffix}.zip"
        save_path = self.saved_models_dir / filename
        
        # Метод save() из SB3 корректно сохраняет и веса, и состояние VecNormalize
        self.model.save(save_path)
        
        logger.info(f"[AGENT] ФИНАЛЬНАЯ МОДЕЛЬ СОХРАНЕНА: {save_path}")

    def save(self):
        """
        Сохраняет текущее состояние модели как промежуточный чекпоинт в папку checkpoints.
        Этот метод можно вызывать вручную (например, при достижении рекорда).
        """
        # Находим следующий свободный номер шага
        existing_checkpoints = list(self.checkpoints_dir.glob("rl_model_*.zip"))
        
        if not existing_checkpoints:
            next_step = 5000 # Начальный шаг, если чекпоинтов еще нет
        else:
            def get_step(p):
                match = re.search(r'_(\d+)_steps', str(p.name))
                return int(match.group(1)) if match else 0
            last_step = max(existing_checkpoints, key=get_step)
            next_step = get_step(last_step) + 5000 # Шаг +5000

        save_path = self.checkpoints_dir / f"rl_model_{next_step}_steps.zip"
        
        self.model.save(save_path)
        
        logger.info(f"[AGENT] ПРОМЕЖУТОЧНЫЙ ЧЕКПОИНТ СОХРАНЕН: {save_path}")