import os
import json
import math

def find_nan_in_json_files(pth):
    """
    Проверяет все JSON файлы в каталоге pth на наличие NaN в свойстве "loss",
    являющемся массивом чисел.  Выводит в консоль названия файлов,
    в которых обнаружены NaN.

    Args:
        pth: Путь к каталогу с JSON файлами.
    """

    for filename in os.listdir(pth):
        if filename.endswith(".json"):
            filepath = os.path.join(pth, filename)
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)

                if "loss" in data and isinstance(data["loss"], list):
                    for value in data["loss"]:
                        if isinstance(value, (int, float)) and math.isnan(value):
                            print(f"{pth}{filename}")
                            break  # Не выводить имя файла несколько раз, если NaN найден
            except json.JSONDecodeError:
                print(f"Ошибка: Не удалось декодировать JSON в файле {filename}")
            except Exception as e:
                print(f"Ошибка при обработке файла {filename}: {e}")


if __name__ == "__main__":
    pth = 'pth/'  
    find_nan_in_json_files(pth)