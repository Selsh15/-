#Импортируются  библиотеки для работы с данными: NumPy для численных вычислений, Pickle для загрузки данных
import numpy as np
import pickle
import os

def load_data(): # Определяется функция load_data(), которая загружает и возвращает датасет CIFAR-10

    if not os.path.exists('cifar-10-python.tar.gz'): # Проверяется, существует ли файл с данными CIFAR-10 в текущей директории. Если нет, то данные будут загружены.
        print('Downloading CIFAR-10 dataset...') # Выводится сообщение о загрузке данных.
    with open('cifar-10-python/data_batch_1', 'rb') as f: # Открывается файл с первой частью обучающих данных в режиме чтения.
        data = pickle.load(f, encoding='bytes') # Загружаются данные из файла с помощью модуля Pickle.
    x_train = data[b'data'] # Извлекаются обучающие изображения из загруженных данных
    y_train = data[b'labels'] # Извлекаются метки классов для обучающих изображений.
    with open('cifar-10-python/test_batch', 'rb') as f: # Открывается файл с тестовыми данными в режиме чтения.
        data = pickle.load(f, encoding='bytes')  # Загружаются тестовые данные из файла.
    x_test = data[b'data'] # Извлекаются тестовые изображения из загруженных данных.
    y_test = data[b'labels'] # Извлекаются метки классов для тестовых изображений.

    return (x_train, y_train), (x_test, y_test) # Возвращается кортеж с обучающими и тестовыми данными.
