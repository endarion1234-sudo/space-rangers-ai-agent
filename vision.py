import logging
from ultralytics import YOLO
import logging
import torch
logger = logging.getLogger(__name__)

class VisionSystem:
    """
    Система зрения. Исправлена для работы с моделью, у которой имена классов - это числа.
    """
    def __init__(self, model_path, pilot_instance, screen_processor):
        self.model_path = model_path
        self.pilot = pilot_instance
        self.screen_processor = screen_processor
        torch.device("cpu")
        self.model = YOLO(self.model_path, verbose=True)
        
             
    def get_game_state(self, frame):
        """
        Анализирует кадр и возвращает список объектов.
        Оптимизированная и чистая версия.
        Использует встроенные имена классов модели.
        """
        if frame is None or self.model is None:
            return None

        try:
            # 1. Вызываем модель БЕЗ лишней конвертации цветов.
            # Кадр от D3DShot уже в нужном формате (BGR или RGB), 
            # модель сама разберется.
            results = self.model(frame)

            # 2. Быстрая проверка на наличие результатов
            if not results or not results[0].boxes.data:
                return None

            objects_found = []

            # 3. Перебираем результаты
            for det in results[0].boxes.data:
                # det содержит тензор с данными [x1, y1, x2, y2, conf, cls_id]
                x1, y1, x2, y2, conf, cls_id = det.tolist()

                if conf < 0.5:
                    continue

                # 4. ГЛАВНОЕ ИЗМЕНЕНИЕ: Получаем имя класса напрямую из модели
                # self.model.model.names — это встроенный список имен классов.
                # cls_id (индекс) -> имя класса.
                try:
                    class_name = self.model.model.names[int(cls_id)]
                except (IndexError, ValueError, TypeError):
                    # Если имя получить не удалось (например, ID вне диапазона),
                    # используем ID как строку для стабильности.
                    class_name = str(cls_id)

                x_center = int((x1 + x2) / 2)
                y_center = int((y1 + y2) / 2)

                objects_found.append({
                    'type': class_name,   # Теперь здесь понятное имя!
                    'coords': (x_center, y_center)
                })

            return objects_found if objects_found else None

        except Exception:
            # В случае ошибки просто возвращаем None, чтобы не прерывать обучение.
            return None
          
    def get_observations(self):
        """
        Сканирует экран и возвращает список обнаруженных объектов.
        Оптимизированная версия: без логирования и отладки для максимальной скорости.
        """
        observations = []
        # Получаем кадр. Если игра свернута, возвращаем пустой список.
        frame = self.screen_processor.get_frame_for_yolo()
        if frame is None:
            return observations

            # Вызов модели. 'stream=False' для точности.
        results = self.model(frame, stream=False)

            # Проверка на наличие результатов
        if results and len(results) > 0 and len(results[0].boxes.data) > 0:
                # Обрабатываем только первый пакет результатов (batch)
                for det in results[0].boxes.data:
                    x1, y1, x2, y2, conf, cls_id = det.tolist()

                    # Фильтрация по порогу уверенности
                    if conf < 0.5:
                        continue

                    # Безопасное получение имени класса
                    try:
                        class_name = self.model.model.names[int(cls_id)]
                    except Exception:
                        continue

                    # Вычисление центра бокса
                    x_center = int((x1 + x2) / 2)
                    y_center = int((y1 + y2) / 2)

                    # Добавление объекта в итоговый список
                    observations.append({
                        'name': class_name,
                        'center': (x_center, y_center),
                        'confidence': float(conf)
                    })
        return observations