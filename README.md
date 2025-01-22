<p align="center">
      <img src='https://i.postimg.cc/kG36d61k/temp-Image-K2l1l-B.avif' alt="Логотип проекта" width="400">
</p>

<p align="center">
   <img src="https://img.shields.io/badge/Python-3.10.14-brightgree" alt="Версия Python">
   <img src="https://img.shields.io/badge/version-v0.1.1-blue" alt="Версия проекта">
   <img src="https://img.shields.io/github/last-commit/ZakatZakat/Digital-Assistant" alt="Последний коммит">
</p>

# Digital Twin LLM Assistant

## О проекте

Этот проект посвящен созданию цифрового двойника LLM-ассистента, разработанного для управления портфелями состоятельных клиентов. Ассистент использует современные языковые модели (LLM), способные анализировать финансовые данные, предлагать инвестиционные стратегии и предоставлять рекомендации, адаптированные к уникальным целям и профилю риска каждого клиента.

## Установка

Сначала необходимо клонировать репозиторий и установить его зависимости.
```bash
1. git clone https://github.com/ZakatZakat/Digital-Assistant.git
2. python setup.py install
```
Затем все модели для проекта должны быть загружены из Ollama.
```bash
1. python pull_models.py
```

## Использование

Используйте скрипт **src/preprocessing.py** для предварительной обработки ваших PDF-файлов.

Чтобы запустить все модели на данных QA, используйте src/run_models.py.

Настройте параметры для run_models через config.yaml:

```yaml
## Доступные модели:
## Доступные модели: 'llama3.1:8b-instruct-q8_0', 'llama3.1:latest', 'gemma2:9b', 'mistral-nemo:12b'

Model: llama3.1:latest  # Указывает модель для использования. Если указано "All", будут загружены все доступные модели.
Questions: 'Default'    # Тип вопросов из словаря вопросов, например 'Default' или 'All'.
Сhain_type: 'refine'    # Тип цепочки для обработки задач. Здесь используется "refine" для уточнения ответа.
Temperature: 0.2        # Настройка температуры модели, влияющая на случайность генерируемого текста. Более низкие значения (0.1-0.3) делают ответы более детерминированными.
```

**content/global_js.json** — файл с очищенными данными для системы RAG.

**content/global_function.json** — файл с необработанными данными для системы RAG (должен обновляться с новыми LLM-моделями).

## Docker

Сборка и запуск сервиса в Linux:
Перед запуском скрипта установки необходимо установить Docker и Ollama локально.

```bash
sudo systemctl start ollama
sudo docker build -t v1 .
sudo docker run -p 8080:8080 v1
```

Использование:
Отправьте запрос на localhost:8080/message с JSON. Поля JSON:
- user_id: целочисленное значение, не обрабатывается
- message: текст запроса, содержащий вопрос или описание

Минимальный пример на Python:
```python
import requests

response = requests.post(
   url='http://127.0.0.1:8080/message',
   json={ 
      "user_id": 0,
      "message": "Как установить ВТБ онлайн на Андроид?",
   }
)
print(response.content.decode())
```

## Документация

TODO

## Разработчики

- [Аскар Ембулатов](https://github.com/ZakatZakat)
- [Фёдор Мельник (бывш)](https://github.com/bezro)
- [Даниил Адаменко](https://github.com/Adam14b)
