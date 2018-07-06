# Detectors-On-Face-Landmarks

## Зависимости
Для работы необходимо скачать предтренированную модель *shape predictor* по [ссылке](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)

Зависимости библиотек python описаны в файле *requirements.txt*, которые можно установить с помощью *pip*: `pip install -r requirements.txt`

Если библиотека *dlib* не устанавливается через *pip* (такое происходит на Windows), то можно скачать Python Wheel file (.whl) по [cсылке](https://pypi.org/simple/dlib/) и установить `pip intall dlib_file.whl`

## Датасет
Для обучения детектора улыбки использовался датасет GENKI-4K, в котором размечены нейтральные лица и с улыбкой

[Описание](http://mplab.ucsd.edu/wordpress/?page_id=398)

[Архив с датасетом](http://mplab.ucsd.edu/wordpress/wp-content/uploads/genki4k.tar)

## Использование
#### Создание лэндмарков
___
Файл `create_landmarks.py` создает папку с текстовыми файлами для каждого изображения, в которых записаны данные о *face landmarks*
```
python create_landmarks.py shape_predictor_file.dat .\genki4k\files .\genki4k\landmarks
```
Аргументы:
- `predictor_file` - файл с *shape predictor*
- `images_path` - путь до папки с изображениями
- `landmarks_path` - путь до директории, в которую будут сохраняться файлы с *face landmarks*

#### Создание модели детектора улыбок
Файл `create_model_smile.py` создает файл с моделью детектора улыбок по размеченному датасету GENKI-4K
```
python create_model_smile.py .\genki4k\labels.txt .\genki4k\landmarks smile_model.pkl
```
Аргументы:
- `labels_file` - файл с разметкой датасета
- `landmarks_path` - пусть до директории с *face landmarks*
- `model_file` - файл, в который сохранится модель

#### Использование детекторов
Файл `use.py` создает файлы, в которые записаны имена изображений, которые удовлетворяют детектору открытого рта и детектору улыбки
```
python use.py shape_predictor_file.dat .\example_data\images smile_model.pkl mouth.txt smile.txt 
```
Аргументы:
- `predictor_file` - файл с *shape predictor*
- `images_path` - путь до папки с изображениями
- `smile_model_file` - файл с моделью детектора улыбок
- `open_mouth_file` - файл, в который запишутся имена изображений, на которых есть открытый рот
- `smile_file` - файл, в который запишутся имена изображений, на которых есть улыбка


