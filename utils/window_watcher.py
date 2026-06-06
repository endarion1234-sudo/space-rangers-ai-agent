import time
import logging
import pygetwindow as gw  # Импортируем как gw для краткости

logger = logging.getLogger("WINDOW")

class WindowWatcher:   
    """  Класс для ожидания готовности окна приложения.
         Блокирует выполнение программы, пока окно не станет активным.
    """


    def __init__(self, window_title, timeout=30, check_interval=1.0):
        self.window_title = window_title
        self.timeout = timeout
        self.check_interval = check_interval

    def wait_for_active(self):
        """
        Блокирует выполнение, пока окно не станет активным или не истечет таймаут.
        :raises: TimeoutError, если окно не стало активным в течение таймаута.
        """
        start_time = time.time()

        while True:
            try:
                # Получаем список всех окон (это основной метод pygetwindow)
                all_windows = gw.getAllWindows()
                
                # Проверяем каждое окно
                for win in all_windows:
                    # Игнорируем окна без заголовка и те, что не свернуты
                    if win.title and not win.isMinimized:
                        # Используем 'in' для поиска подстроки в заголовке
                        if self.window_title in win.title:
                            elapsed_time = round(time.time() - start_time, 1)
                            logger.debug(f" Окно '{self.window_title}' активно! (Ждал {elapsed_time} секунд). Разблокирую выполнение.")
                            return True

                # Если цикл for закончился и мы здесь, значит нужное окно не найдено среди активных
                logger.debug(f"Окно с '{self.window_title}' еще не активно. Следующая проверка через {self.check_interval} сек...")

            except Exception as e:
                logger.warning(f"Ошибка при получении списка окон: {e}")

            # Проверка на таймаут
            if self.timeout > 0 and (time.time() - start_time) > self.timeout:
                elapsed_time = round(time.time() - start_time, 1)
                error_msg = f" ТАЙМАУТ: Окно '{self.window_title}' не стало активным за {elapsed_time} секунд."
                logger.error(error_msg)
                raise TimeoutError(error_msg)

            time.sleep(self.check_interval)