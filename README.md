# roc_terminal
Функция генерации графика Roc-Кривой из терминала.
## Список улучшений
1. Заменил функцию np.dot на np.einsum (для более быстрого подсчета similarity)
2. Избавился от лишних циклов при создании массива данных для roc_curve.
3. Оптимизировал поиск T.

## Установка
1. Перейдите в директорию с проектом и выполните:
```console
user@ubuntu: git clone https://github.com/felixo/roc_terminal
```

2. Перейдите в директорию с env, создайте окружение под проект:
```console
user@ubuntu: virtualenv -p python3 roc_terminal
```
активизируйте env
```console
user@ubuntu: . /path/to/env/roc_terminal/bin/activate
```
перейдите в директорию проекта roc_terminal и выполните:
```console
user@ubuntu: pip install requirements.txt
```

## Использование
Для удобства перекиньте feature_file и person_id_file в директорию проекта (предоставленные наборы уже есть в директории проекта).
Чтобы создать график воспользуйтесь командой:
```console
user@ubuntu: python main.py -f <featurefile> -p <personfile>
```

Например:
```console
user@ubuntu: python main.py -f features.npy -p person_id.npy
```
