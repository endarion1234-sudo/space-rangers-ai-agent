import numpy as np
import cv2
import d3dshot 
             
class ScreenProcessor:
    def __init__(self, game_region, cfg):
        """
        Инициализация процессора.
        Здесь мы создаем объект захвата экрана (D3DShot).
        """
        self.game_region = game_region
        self.cfg = cfg
        self.screen_w = cfg['IMG_WIDTH'] 
        self.screen_h = cfg['IMG_HEIGHT']
        self.d3d = d3dshot.create(capture_output="numpy", frame_buffer_size=30)

    def _get_raw_screenshot(self):
        """
        Возвращает кадр в формате BGR (OpenCV).
        Пересоздает объект D3DShot каждые N кадров для очистки памяти.
        """
        frame = self.d3d.screenshot(region=self.game_region)
        return frame
 
    def _process_frame_for_cv(self, frame):#этот метод нужно убрать он инвертирует 
        """
        Минимальная обработка кадра ТОЛЬКО для CV.
        Убраны все операции записи на диск и логирования.
        """
        return cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
            
    def get_state(self, cursor_position=None):
        """
        ВОЗВРАЩАЕТ ОДИН ОБРАБОТАННЫЙ КАДР.
        """
        raw_frame = self._get_raw_screenshot()
        resized = cv2.resize(
            raw_frame,
            (self.cfg['IMG_WIDTH'], self.cfg['IMG_HEIGHT']),
            interpolation=cv2.INTER_AREA
        )
        return resized
    
    def get_screen(self):
        """
        Метод для совместимости с VisionSystem.
        Просто вызывает основной метод получения состояния.
        """
        print("[SCREEN] Метод get_screen() ВЫЗВАН (для совместимости).")
        # Вызываем основной метод, передавая None вместо координат курсора
        return self.get_state(cursor_position=None)
    

    def get_current_frame_for_cv(self):
        """
        Возвращает кадр для CNN
        """
        raw_frame = self._get_raw_screenshot()
        resized_frame = cv2.resize(
        raw_frame, 
        (self.cfg['IMG_WIDTH'], self.cfg['IMG_HEIGHT']), 
        interpolation=cv2.INTER_AREA
    )
        return resized_frame
        
    def close(self):
        """
        Явный метод для освобождения ресурсов.
        Вызывается при завершении работы скрипта.
        """
        if hasattr(self, 'd3d') and self.d3d is not None:
            self.d3d.stop()
            print("[SCREEN] D3DShot остановлен (явно).")   
    
    # В файле screen_processor.py

    def get_frame_for_yolo(self):
        """
        Возвращает кадр, подготовленный ИСКЛЮЧИТЕЛЬНО для модели YOLO (Ultralytics).
        1) Изменяет размер до 1920x1080 для ускорения работы на CPU.
        2) Оставляет цвета в RGB, так как Ultralytics YOLO ожидает этот формат.
        """
        raw_frame = self._get_raw_screenshot()

        if raw_frame is None:
                return None
        rgb_frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
      
        resized_frame = cv2.resize(rgb_frame, (1280, 720), interpolation=cv2.INTER_AREA)
        
        return resized_frame 
    
    def __del__(self):
        """
        Этот метод вызывается при уничтожении объекта.
        Он гарантирует, что захват экрана будет остановлен,
        даже если в основном коде про это забыли.
        """
        print("[DEBUG] [SCREEN_PROCESSOR] Вызван деструктор. Останавливаем D3DShot.")
        try:
            # Проверяем, существует ли объект d3d и есть ли у него метод stop
            if hasattr(self, 'd3d') and self.d3d is not None:
                self.d3d.stop()
        except Exception as e:
            print(f"[WARNING] Ошибка при остановке D3DShot в деструкторе: {e}")
