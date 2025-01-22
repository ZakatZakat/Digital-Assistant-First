import os
import yaml
from dotenv import load_dotenv
from huggingface_hub import login
from generation import *
from QA import *
from utils.paths import ROOT_DIR

def load_config(config_path):
    """Загрузить конфигурацию из YAML файла."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_env_variables():
    """Загрузить переменные окружения из файла .env."""
    load_dotenv()
    return os.getenv("HF_API_TOKEN")

def authenticate_huggingface(token):
    """Аутентификация в Hugging Face с использованием предоставленного токена."""
    if not token:
        raise ValueError("HF_API_TOKEN не установлен. Проверьте ваш .env файл.")
    login(token=token)

def process_model(model, temperature, questions):
    """Выполнить инференс для каждой модели и вопроса."""
    if not questions:
        raise ValueError("QA_dictinary пуст. Нет вопросов для обработки.")
    
    for question in questions:
        get_inference(model=model, temperature=temperature, query=question)

def main():
    config = load_config(ROOT_DIR / 'config.yaml')

    # Загрузка и аутентификация в Hugging Face
    token = load_env_variables()
    authenticate_huggingface(token)

    # Получение параметров конфигурации
    model = config.get('Model')
    temperature = config.get('Temperature')
    question_type = config.get('Questions')

    if not model:
        raise KeyError("Ключ 'Model' не найден в файле конфигурации или он пуст.")

    # Определить, какие модели нужно обработать
    models_to_process = ['llama3.1:8b-instruct-q8_0', 'llama3.1:latest', 'gemma2:9b', 'mistral-nemo:12b'] if model == 'All' else [model]
    
    # Обработать каждую модель с предоставленными вопросами
    if question_type:
        questions = QA_dictinary.get(question_type, {}).keys()
        for model in models_to_process:
            process_model(model, temperature, questions)
    else:
        raise KeyError("Ключ 'Questions' не найден в файле конфигурации или он пуст.")

if __name__ == "__main__":
    main()
    