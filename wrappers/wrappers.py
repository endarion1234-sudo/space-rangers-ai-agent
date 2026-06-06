import gymnasium as gym
import numpy as np

class DictifyObservation(gym.ObservationWrapper):
    """
    Обертка, которая превращает векторное наблюдение (Box)
    в словарное (Dict), которое ждет MultiInputPolicy.
    """
    def __init__(self, env):
        super().__init__(env)
        
        # Сохраняем старое пространство (вектор)
        self.original_space = self.env.observation_space
        
        # Создаем НОВОЕ пространство (словарь)
        # Это нужно, чтобы агент знал, что его ждет.
        self.observation_space = gym.spaces.Dict({
            'data': self.original_space
        })
        print(f"[WRAPPER] Пространство изменено на Dict. Вектор: {self.original_space.shape}")

    def observation(self, obs):
        """
        Этот метод вызывается автоматически.
        Он принимает вектор 'obs' и возвращает словарь.
        """
        # Логируем для отладки (можно убрать)
        # print(f"[WRAPPER] Получили вектор формы: {obs.shape}")
        
        # Возвращаем словарь, который ждет MultiInputPolicy
        return {'data': obs}