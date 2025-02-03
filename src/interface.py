#Импорты стандартной библиотеки
import logging
import json
import tempfile
import pymupdf
import os
import yaml
import pandas as pd
import streamlit as st
import asyncio

# Импорты сторонних библиотек
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from src.utils.check_serp_response import APIKeyManager
from src.offergen.agent import validation_agent
from src.offergen.utils import get_system_prompt_for_offers
from src.utils.logging import setup_logging, log_api_call
from src.internet_search import *

# Локальные импорты
from src.utils.kv_faiss import KeyValueFAISS
from src.utils.paths import ROOT_DIR

logger = setup_logging(logging_path='logs/digital_assistant.log')

serpapi_key_manager = APIKeyManager(path_to_file="api_keys_status.xlsx")

def load_config_yaml(config_file="config.yaml"):
    """Загрузить конфигурацию из YAML-файла."""
    with open(config_file, "r", encoding="utf-8") as f:
        config_yaml = yaml.safe_load(f)
    return config_yaml

def model_response_generator(retriever, model, config):
    """Сгенерировать ответ с использованием модели и ретривера."""
    config_yaml = load_config_yaml()
    user_input = st.session_state["messages"][-1]["content"]

    try:
        # Формирование истории сообщений (исключая системное сообщение)
        message_history = ""
        if "messages" in st.session_state and len(st.session_state["messages"]) > 1:
            history_messages = [
                f"{msg['role']}: {msg['content']}"
                for msg in st.session_state["messages"]
                if msg.get("role") != "system"
            ]
            history_size = int(config.get("history_size", 0))
            if history_size:
                history_messages = history_messages[-history_size:]
            message_history = "\n".join(history_messages)
        
        # Если интернет-поиск включён, вызываем функции поиска, иначе возвращаем пустую строку
        if config_yaml.get("internet_search", False):
            _, serpapi_key = serpapi_key_manager.get_best_api_key()

            shopping_res = search_shopping(user_input, serpapi_key)
            internet_res, links, coordinates = search_places(user_input, serpapi_key)
            maps_res = search_map(user_input, coordinates, serpapi_key)
            yandex_res = yandex_search(user_input, serpapi_key)
        else:
            shopping_res = ""
            internet_res = ""
            links = ""
            maps_res = ""
            yandex_res = ""
        
        # Если система работает в режимах RAG или File
        if config['System_type'] in ['RAG', 'File']:
            
        
            # Загрузка системного промпта из YAML-конфига
            system_prompt_template = config_yaml["system"]["system_prompt"]
            
            # Форматирование промпта с подстановкой переменных
            formatted_prompt = system_prompt_template.format(
                context=message_history,
                internet_res=internet_res,
                links=links,
                shopping_res=shopping_res,
                maps_res=maps_res,
                yandex_res=yandex_res
            )

            # Создание цепочки для модели, если имя модели начинается с 'gpt'
            if config['Model'].startswith('gpt'):
                
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                # Run validation in the event loop
                # other wise it will be blocked by the main thread and fail UI
                offer_validation_response = loop.run_until_complete(
                    validation_agent.run(user_input)
                ).data
                logger.info(f"Offer validation response: {offer_validation_response}")
                if not offer_validation_response.is_valid:
                    logger.info("Offer validation failed")
                    prompt = ChatPromptTemplate.from_messages([
                        ("system", formatted_prompt),
                        ("human", "User query: {input}\nAdditional context: {context}")
                    ])
                else:
                    logger.info("Offer validation passed")
                    system_prompt = get_system_prompt_for_offers(offer_validation_response, user_input)
                    prompt = ChatPromptTemplate.from_messages([
                        ("system", system_prompt),
                        ("human", "User query: {input}\nAdditional context: {context}")
                    ])
    
                question_answer_chain = create_stuff_documents_chain(model, prompt)
                rag_chain = create_retrieval_chain(retriever, question_answer_chain)

                full_answer = ""
                for chunk in rag_chain.stream({"input": user_input}):
                    if "answer" in chunk:
                        full_answer += chunk["answer"]
                        yield {"answer": chunk["answer"], "maps_res": maps_res}

            log_api_call(
                    logger=logger,
                    source=f"LLM ({config['Model']})",
                    request=user_input,
                    response=full_answer,
                )
    except Exception as e:
        log_api_call(
            logger=logger,
            source=f"LLM ({config['Model']})",
            request=user_input,
            response="",
            error=str(e)
        )
        raise

def handle_user_input(retriever, model, config):
    """Обработать пользовательский ввод и сгенерировать ответ ассистента."""
    prompt = st.chat_input("Введите запрос здесь...")
    if prompt:
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            response_text = ""
            maps_res = []  # Инициализируем maps_res
            for chunk in model_response_generator(retriever, model, config):
                response_text += chunk["answer"]
                response_placeholder.markdown(response_text)
                if isinstance(chunk.get("maps_res"), list):
                    maps_res = chunk["maps_res"]

            st.session_state["messages"].append(
                {"role": "assistant", "content": response_text, "question": prompt}
            )

               # Проверка и обработка maps_res
        
def init_message_history(template_prompt):
    """Инициализировать историю сообщений для чата."""
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
        with st.chat_message('System'):
            st.markdown(template_prompt)
        


def display_chat_history():
    """Отобразить историю чата из состояния сессии."""
    for message in st.session_state["messages"][1:]:
        with st.chat_message(message["role"]):
            if 'question' in message:
                st.markdown(message["question"])
            st.markdown(message['content'])


def creating_documents(config):
    """Создать векторное пространство на основе конфигурации."""
    if config["System_type"] == 'File' and 'Uploaded_file' in config and config['Uploaded_file'] is not None:
        uploaded_file = config['Uploaded_file']
        # Обработка загруженного файла
        if uploaded_file.type == "text/plain":
            # Чтение текстового файла
            file_content = uploaded_file.getvalue().decode("utf-8")
            documents = [Document(page_content=file_content)]
            return documents
        
        elif uploaded_file.type == "application/pdf":
       # Чтение PDF файла с использованием PyMuPDF
            # Сохранение загруженного файла во временный файл
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            # Используем PyMuPDF для извлечения текста из PDF
            with pymupdf.open(os.path.normpath(tmp_file_path)) as doc:
                text = chr(12).join([page.get_text() for page in doc])
                paragraphs = text.split('\x0c')
                formatted_paragraphs = [
                    ' '.join(paragraph.replace('\n', ' ').replace('\xa0', ' ').split())
                    for paragraph in paragraphs
                ]
                file_content = '\n\n'.join(formatted_paragraphs)
                documents = [Document(page_content=file_content)]

            # Удаление временного файла
            os.unlink(tmp_file_path)
            return documents

        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            # Чтение DOCX файла
            from langchain_community.document_loaders import Docx2txtLoader
            loader = Docx2txtLoader(tmp_file_path)
            documents = loader.load()
            return documents
        else:
            st.error("Неподдерживаемый тип файла.")
            return None
    else:
        path = ROOT_DIR / 'content' / 'json' / 'pp_data.json'
        loader = TextLoader(path)
        documents = loader.load()
        
        return documents
    
def create_retriever(splitter_type, embeddings_model, documents, chunk_size):
    """Создать ретривер для извлечения данных."""
    if splitter_type == 'character':
        embeddings = (
            HuggingFaceEmbeddings(model_name=embeddings_model)
            if embeddings_model != 'OpenAIEmbeddings'
            else OpenAIEmbeddings(model='text-embedding-ada-002')
        )
        splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
        docs = splitter.split_documents(documents)
        vector_store = FAISS.from_documents(docs, embeddings)
    
        return vector_store
    
    elif splitter_type == 'json':
        embeddings = (
            HuggingFaceEmbeddings(model_name=embeddings_model)
            if embeddings_model != 'OpenAIEmbeddings'
            else OpenAIEmbeddings(model='text-embedding-ada-002')
        )
        logical_chunks = json.loads(documents[0].page_content)
        kv_dict = {k: f'{k}:\n{v}' for k, v in logical_chunks.items()}
        docs = [Document(k) for k in logical_chunks.keys()]
        vector_store = KeyValueFAISS.from_documents(docs, embeddings).add_value_documents(kv_dict)
        return vector_store

    else:
        raise ValueError(f"Неподдерживаемый тип разбиения: {splitter_type}")

def create_vector_space(config):
    """Создать векторное пространство для извлечения документов на основе конфигурации."""

    embeddings_model = config['Embedding']
    splitter_type = config['Splitter']['Type']
    chunk_size = int(config['Splitter']['Chunk_size'])

    
    documents = creating_documents(config)

    vector_store = create_retriever(splitter_type, embeddings_model, documents, chunk_size)
    
    return vector_store
