# Файл: custom_extractor.py

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium.spaces import Dict

class CustomVecExtractor(BaseFeaturesExtractor):
    """
    Кастомный экстрактор для склеенного вектора [data + mask].
    Принимает на вход словарь {'vec_features': склеенный_вектор, ...}.
    """

    def __init__(self, observation_space: Dict, features_dim: int = 128):
        # Размерность ВЫХОДА экстрактора. 
        # Мы не сохраняем features_dim как self.features_dim!
        output_dim = features_dim 

        # Размерность ВХОДА (input_dim) мы вычисляем сами
        input_dim = observation_space['vec_features'].shape[0] 
        
        # Инициализируем базовый класс. Обратите внимание: мы передаем 0,
        # потому что мы сами вычислим размерность позже.
        super(CustomVecExtractor, self).__init__(observation_space, features_dim=0)

        # Нейросеть для обработки данных
        self.net = nn.Sequential(
            nn.Linear(input_dim // 2, output_dim), # Принимаем только половину вектора (data)
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.ReLU()
        )
        
        # Сохраняем размерность ВЫХОДА как свойство
        self._features_dim = output_dim 

    @property
    def features_dim(self) -> int:
        """Геттер для размера выходного вектора."""
        return self._features_dim

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Принимает склеенный вектор и разделяет его.
        """
        # 1. Получаем склеенный вектор из словаря
        combined_vector = observations['vec_features']
        
        # 2. Разделяем его на данные и маску
        # Мы берем только первую половину (сами данные), так как маска уже применена при подготовке данных в Env.
        # Или, если хотите применить маску здесь:
        mid_point = combined_vector.shape[-1] // 2
        data = combined_vector[..., :mid_point]
        mask = combined_vector[..., mid_point:]
        
        filtered_data = data * mask 
        
        # 3. Прогоняем через нейросеть
        return self.net(filtered_data)