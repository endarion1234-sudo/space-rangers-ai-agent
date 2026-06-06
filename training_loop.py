# В файле training_loop.py

# 1. Убираем TensorBoardCallback из импорта
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from callbacks import RewardForDistanceCallback 

def run_training(agent, cfg):
    total_steps = cfg.get('TOTAL_TIMESTEPS', 50000)
    save_path = agent.checkpoints_dir

    # --- 1. Коллбэк для сохранения чекпоинтов ---
    checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path=save_path,
        name_prefix="rl_model",
        verbose=1,
    )

    # --- 2. Ваш коллбэк для расчета награды ---
    reward_callback = RewardForDistanceCallback()

    # --- 3. Объединяем коллбэки в список ---
    # Обратите внимание: мы больше не добавляем "tb_callback"
    callback = CallbackList([
        checkpoint_callback,
        reward_callback,
    ])

    agent.get_model().learn(
        total_timesteps=total_steps,
        callback=callback, 
        log_interval=1000, # Этот параметр теперь важнее, чем раньше
        reset_num_timesteps=False,
    )
    
    agent.save_final_model()
    print("[TRAIN] Обучение завершено.")
    return True