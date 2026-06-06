import gymnasium as gym
import numpy as np
from gymnasium import spaces
import logging

logger = logging.getLogger("WRAPPER_DEBUG")

class VecEnvFix(gym.Wrapper):
    """
    Хирургическая обертка для исправления несовместимости API Gym.
    Гарантирует корректный возврат данных из reset и step.
    """
    def __init__(self, env):
        super().__init__(env)
        self._last_info = None
        self.num_envs = 1 

    def reset(self, **kwargs):
        """
        Исправляет возврат из reset().
        Гарантирует, что всегда возвращается кортеж (obs, info).
        """
        result = self.env.reset(**kwargs)
        
        # --- НОВАЯ ЛОГИКА: Проверяем тип результата ---
        # Если результат - это кортеж из 2 элементов, считаем что это (obs, info)
        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
            self._last_info = info
            return obs # Возвращаем только obs, как ожидает VecEnv
        
        # Если результат - это только obs (вектор или кадр)
        # Оборачиваем его в кортеж с пустым словарем info
        obs = result
        info = self._last_info or {}
        self._last_info = info
        return obs, info # Возвращаем кортеж для совместимости со step

    def step(self, action):
        result = self.env.step(action)
        
        if isinstance(result, tuple) and len(result) == 5:
            return result
            
        if isinstance(result, tuple) and len(result) == 3:
            obs, reward, done = result
            return obs, reward, done, done, self._last_info or {}
            
        return result