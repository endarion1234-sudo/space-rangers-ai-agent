from stable_baselines3.common.callbacks import BaseCallback
import os
class RewardForDistanceCallback(BaseCallback):
    """
    Коллбэк для динамического расчета награды и вывода итогов.
    """

    def __init__(self, log_path="C:/space_ai/training_log_1.txt", verbose=0):
        super().__init__(verbose)
        self._last_distance = None
        self._episode_total_reward = 0.0
        self.log_path = log_path

        # Создаем директорию, если её нет
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

        # Открываем файл в режиме добавления (append)
        # 'a' - append: добавляет новые строки в конец файла, не удаляя старые
        self.log_file = open(self.log_path, 'a', encoding='utf-8')
        
        # Записываем заголовок, если файл новый или хотим зафиксировать время начала
        # self.log_file.write(f"--- Лог обучения начат: {datetime.datetime.now()} ---\n")

        print(f"[DEBUG] [CALLBACK] Коллбэк успешно инициализирован. Лог пишется в {self.log_path}")

    def _on_step(self) -> bool:
        """
        Вызывается на каждом шаге.
        """
        # --- 1. НАКОПЛЕНИЕ НАГРАДЫ ---
        # Получаем текущую награду из локальных переменных
        # 'rewards' - это массив наград (даже если окружение одно)
        if 'rewards' in self.locals:
            # Складываем текущую награду за шаг в общую копилку эпизода
            # Используем .item(), так как награды в SB3 обычно тензоры
            current_reward = self.locals['rewards'][0].item() 
            self._episode_total_reward += current_reward

        # --- 2. ЛОГИКА ЗА СБЛИЖЕНИЕ (остается без изменений) ---
        if 'infos' in self.locals and self.locals['infos']:
            current_infos = self.locals['infos'][0]
            if 'distance_to_target' in current_infos:
                real_distance = current_infos['distance_to_target']
                if self._last_distance is not None:
                    delta = self._last_distance - real_distance
                    if delta > 0:
                        reward_for_distance = min(delta * 10, 15.0)
                        # Награда уже применена к self.locals['rewards']
                        print(f"[DEBUG] [REWARD] Награда за сближение: {reward_for_distance:+.3f}")
                    elif delta < 0:
                        penalty_for_distance = max(delta * 0.5, -2.0)
                        print(f"[DEBUG] [PENALTY] Штраф за удаление: {penalty_for_distance:+.3f}")
                self._last_distance = real_distance

        return True 

    def _on_rollout_end(self) -> None:
        """
        Вызывается в конце каждого эпизода. Записывает итоговую награду в файл.
        """
        # Формируем строку для записи
        log_entry = f"Эпизод {self.num_timesteps} | Итоговая награда: {self._episode_total_reward:.2f}\n"
        
        # Пишем в файл
        self.log_file.write(log_entry)
        
        # Сбрасываем счетчик для следующего эпизода
        self._episode_total_reward = 0.0 

    def _on_training_end(self) -> None:
        """
        Вызывается один раз в самом конце всего обучения.
        Здесь мы закрываем файл, чтобы сохранить все данные и освободить ресурс.
        """
        print(f"[DEBUG] [CALLBACK] Обучение завершено. Закрытие лог-файла.")
        self.log_file.close()