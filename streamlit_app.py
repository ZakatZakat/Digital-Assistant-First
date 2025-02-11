# Импорты стандартной библиотеки
import logging
import time

# Импорты сторонних библиотек
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Локальные импорты
from src.interface import *
from langchain_core.documents import Document

def setup_logging():
    """Настройка конфигурации логирования."""
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def load_config_yaml(config_file="config.yaml"):
    """Загрузить конфигурацию из YAML-файла."""
    with open(config_file, "r", encoding="utf-8") as f:
        config_yaml = yaml.safe_load(f)
    return config_yaml

def load_available_models():
    """Загрузка доступных моделей из Ollama и добавление пользовательских моделей."""
    #models = [model['name'] for model in ollama.list()['models']]
    #models.extend(['gpt-4o', 'gpt-4o-mini'])
    models = ['gpt-4o', 'gpt-4o-mini']
    return models


def initialize_session_state(defaults):
    """Инициализация состояния сессии с использованием значений по умолчанию."""
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def apply_configuration():
    """Применить выбранную конфигурацию и обновить состояние сессии."""
    config = {
        "Model": st.session_state["selected_model"],
        "Chain_type": st.session_state["selected_chain_type"],
        "System_type": st.session_state["selected_system"],
        "Temperature": st.session_state["selected_temperature"],
        "Embedding": st.session_state["selected_embedding_model"],
        "Splitter": {
            "Type": st.session_state["selected_splitter_type"],
            "Chunk_size": st.session_state["chunk_size"],
        },
        "history": st.session_state["history"],
        "history_size": st.session_state["history_size"],
    }

    if st.session_state["selected_system"] == 'File' and st.session_state.get("uploaded_file") is not None:
        config['Uploaded_file'] = st.session_state["uploaded_file"]

    st.session_state['config'] = config
    st.session_state['config_applied'] = True
    time.sleep(2)
    st.rerun()


def initialize_model(config):
    """Инициализация языковой модели на основе конфигурации."""
    if config["Model"].startswith('gpt'):
        return ChatOpenAI(model=config["Model"], stream=True)
    else:
        # Заглушка для других моделей
        return config["Model"]


def initialize_vector_store(config):
    """Создание векторного пространства на основе конфигурации."""
    return create_vector_space(config)


def display_banner_and_title():
    """Отображение баннера и заголовка."""
    st.image(
        'https://i.ibb.co/yPcRsgx/AMA.png',
        use_container_width=True, width=3000
    )
    st.title("Цифровой Помощник AMA")


def chat_interface(config):
    """Отображение интерфейса чата на основе примененной конфигурации."""
    logger = logging.getLogger(__name__)
    logger.info(f"Конфигурация загружена: {config}")

    template_prompt = "Я ваш Цифровой Ассистент - пожалуйста, задайте свой вопрос."

    if config['System_type'] != 'default':    
        vector_store = initialize_vector_store(config)
    
        if vector_store is None and config['System_type'] != 'default':
            st.error("Не удалось инициализировать векторное хранилище.")
            return

    model = initialize_model(config)

    # Настройка ретривера в зависимости от типа системы
    if config['System_type'] == 'default':
        retriever = None
    elif config['System_type'] in ['RAG', 'File']:
        retriever = vector_store.as_retriever()
    else:
        retriever = None

    init_message_history(template_prompt)
    display_chat_history()
    handle_user_input(retriever, model, config)


def main():
    """Основная функция для запуска приложения Streamlit."""
    load_dotenv()
    logger = setup_logging()
    config_yaml = load_config_yaml()
    # Статичные параметры опций
    options = {
        "models": load_available_models(),
        "system_types": ['RAG'],
        "embedding_models": [
            'OpenAIEmbeddings',
        ],
        "splitter_types": ['character'],
        "chain_types": ['refine'],
        "history": ['Off'],
    }

    defaults = {
        'config_applied': False,
        'config': None,
        'selected_model': config_yaml['model'],
        'selected_system': options["system_types"][0],
        'selected_chain_type': options["chain_types"][0],
        'selected_temperature': 0.2,
        'selected_embedding_model': options["embedding_models"][0],
        'selected_splitter_type': options["splitter_types"][0],
        'chunk_size': 2000,
        'history': 'On',
        'history_size': 10, 
        'uploaded_file': None,
    }

    initialize_session_state(defaults)

    mode = st.sidebar.radio("Выберите режим:", ("Чат", "Поиск по картам 2ГИС"))

    # Применяем конфигурацию сразу без выбора
    if not st.session_state['config_applied']:
        apply_configuration()
    else:
        display_banner_and_title()
        if mode == "Поиск по картам 2ГИС":
            st.session_state['config']['mode'] = '2Gis'
            chat_interface(st.session_state['config'])
        else:
            st.session_state['config']['mode'] = 'Chat'
            chat_interface(st.session_state['config'])


if __name__ == "__main__":
    main()
