import cv2
import numpy as np
import os
from ultralytics import YOLO
import matplotlib.pyplot as plt

class IntegratedRangerVision:
    def __init__(self, model_path='yolov8x.pt', names_path='obj.names'):
        if not os.path.exists(names_path):
            print(f" Ошибка: {names_path} не найден!")
            return
        
        self.model = YOLO(model_path)
        with open(names_path, 'r') as f:
            self.classes = [line.strip() for line in f.readlines() if line.strip()]

        self.groups = {
            'Dominators': ['keller', 'blazer', 'terron', 'dominator', 'clig', 'bertor'],
            'Neutrals_Pirates': ['pirate', 'bandit', 'liner', 'transport', 'ranger', 'diplomat', 'military_ship'],
            'Player': ['player'],
            'Stations_Planets': ['planet', 'station', 'base', 'center', 'dominion', 'med_center', 'ranger_centr', 'scientific_station'],
            'Stars': ['star', 'sun', 'black_hole'],
            'Loot': ['container', 'item', 'minerales', 'micromodul', 'asteroide', 'cisterna']
        }

    def get_meta_idx(self, class_id):
        if class_id >= len(self.classes): return -1
        name = self.classes[class_id].lower()
        for i, (group_name, keywords) in enumerate(self.groups.items()):
            if any(key in name for key in keywords): return i
        return -1

    def analyze_frame(self, frame_path, map_size=(64, 64)):
        results = self.model.predict(frame_path, conf=0.3, verbose=False)[0]
        f_map = np.zeros((len(self.groups), map_size[0], map_size[1]), dtype=np.float32)
        
        h, w = results.orig_shape
        for box in results.boxes:
            cls_id = int(box.cls[0])
            meta_idx = self.get_meta_idx(cls_id)
            if meta_idx != -1:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx = int(((x1 + x2) / 2) / w * map_size[1])
                cy = int(((y1 + y2) / 2) / h * map_size[0])
                if 0 <= cx < map_size[1] and 0 <= cy < map_size[0]:
                    f_map[meta_idx, cy, cx] = float(box.conf[0])
        
        return results.plot(), f_map

    def show_all(self, annotated_img, f_map):
        # 1. Показываем детекцию (OpenCV)
        cv2.imshow('YOLO Detection', annotated_img)
        
        # 2. Показываем Feature Map (Matplotlib)
        group_names = list(self.groups.keys())
        fig, axes = plt.subplots(1, len(group_names), figsize=(20, 4))
        fig.canvas.manager.set_window_title('AI Feature Map Layers')
        for i, name in enumerate(group_names):
            axes[i].imshow(f_map[i], cmap='magma')
            axes[i].set_title(name)
            axes[i].axis('off')
        plt.show()
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# --- ЗАПУСК ---
if __name__ == '__main__':
    # Укажите свои пути
    CONFIG = {
        'model': r'C:\CVAT\best_v8x_aug_1024.pt',
        'names': 'blazers/obj.names',
        'image': r'C:\CVAT\train\images\planet_133.png'
    }

    rv = IntegratedRangerVision(CONFIG['model'], CONFIG['names'])
    img, fm = rv.analyze_frame(CONFIG['image'])
    rv.show_all(img, fm)