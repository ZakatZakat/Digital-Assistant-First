#Импорты стандартной библиотеки
import logging
import subprocess
import json
import ollama
import tempfile
import pymupdf
import os
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder
# Импорты сторонних библиотек
import streamlit as st
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from serpapi import GoogleSearch
from src.utils.check_serp_response import APIKeyManager

from src.utils.logging import setup_logging, log_api_call

# Локальные импорты
from src.utils.kv_faiss import KeyValueFAISS
from src.utils.paths import ROOT_DIR

# Настройка логирования
# logging.basicConfig(
#     level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)

logger = setup_logging(logging_path='logs/digital_assistant.log')

serpapi_key_manager = APIKeyManager(path_to_file="api_keys_status.xlsx")
# serpapi_key_name, serpapi_key = serpapi_key_manager.get_best_api_key()
# print('выбран ключ', serpapi_key)
# serpapi_key = '8f7a24637047a7906eb5e0b4780d849c0ef50e0ebac4127091ee45773a3b3f17'

def search_map(q, coordinates):
    try:
        # Проверяем, есть ли координаты и их значения
        if not coordinates or not coordinates.get('latitude') or not coordinates.get('longitude'):
            return []  # Возвращаем пустоту, если координаты отсутствуют
        
        latitude = coordinates.get('latitude')
        longitude = coordinates.get('longitude')
        zoom_level = "14z"  # Укажите необходимый уровень масштабирования карты

        # Формируем параметр ll из coordinates
        ll = f"@{latitude},{longitude},{zoom_level}"

        # Параметры запроса
        _, serpapi_key = serpapi_key_manager.get_best_api_key()
        params = {
            "engine": "google_maps",
            "q": q,
            "ll": ll,
            "api_key": serpapi_key
        }

        search = GoogleSearch(params)
        results = search.get_dict()

        good_results = [
        [
            item.get('title', 'Нет информации'),
            item.get('rating', 'Нет информации'),
            item.get('reviews', 'Нет информации'),
            item.get('address', 'Нет информации'),
            item.get('website', 'Нет информации'),
            item.get('phone', 'Нет информации'),

        ]
        for item in results.get('local_results', [])
        ]

        log_api_call(
            logger=logger,
            source="SerpAPI Maps",
            request=q,
            response=good_results
        )

        return good_results
    
    except Exception as e:
        log_api_call(
            logger=logger,
            source="SerpAPI Maps",
            request=q,
            response="",
            error=str(e)
        )
        raise


def search_shopping(q):
    try:
        _, serpapi_key = serpapi_key_manager.get_best_api_key()
        params = {
            "engine": "google_shopping",
            "q": q,
            "api_key": serpapi_key
            }

        search = GoogleSearch(params)
        results = search.get_dict()

        results_with_titles_and_links = [
            (item['title'], item['link'])
            for item in results.get('organic_results', [])
            if 'title' in item and 'link' in item
        ]

        log_api_call(
            logger=logger,
            source="SerpAPI Shopping",
            request=q,
            response=results_with_titles_and_links
        )

        return results_with_titles_and_links
    
    except Exception as e:
        log_api_call(
            logger=logger,
            source="SerpAPI Shopping",
            request=q,
            response="",
            error=str(e)
        )
        raise

def search_places(q):
    """Search for places using Google Search API, возвращает только первые 5 результатов."""
    try:
        _, serpapi_key = serpapi_key_manager.get_best_api_key()
        params = {
            "q": q,
            #'location': 'Russia',
            "hl": "ru",
            "gl": "ru",
            "google_domain": "google.com",
            "api_key": serpapi_key
        }

        search = GoogleSearch(params)
        results = search.get_dict()

        good_results = [
        item['snippet']
        for item in results.get('organic_results', [])
        if 'snippet' in item]

        results_with_titles_and_links = [
            (item['title'], item['link'])
            for item in results.get('organic_results', [])
            if 'title' in item and 'link' in item
        ]

        coordinates = results.get('local_map', {}).get('gps_coordinates', None)

        # Укажите путь к папке, где вы хотите сохранить файл
        output_folder = "./"
        output_file = "results.json"

        # Создайте папку, если она не существует
        os.makedirs(output_folder, exist_ok=True)

        # Полный путь к файлу
        output_path = os.path.join(output_folder, output_file)

        # Сохранение данных в файл JSON
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

        print(f"Результаты сохранены в: {output_path}")

        log_api_call(
            logger=logger,
            source="SerpAPI Places",
            request=q,
            response=good_results + results_with_titles_and_links
        )

        return good_results, results_with_titles_and_links, coordinates
    
    except Exception as e:
        log_api_call(
            logger=logger,
            source="SerpAPI Places",
            request=q,
            response="",
            error=str(e)
        )
        raise

def get_ollama_models():
    """Получить список моделей из Ollama."""
    try:
        result = subprocess.run(['ollama', 'list'], stdout=subprocess.PIPE, text=True)
        output = result.stdout.strip()
        models = [line.strip() for line in output.split('\n') if line.strip()]
        return models
    except Exception as e:
        logger.error(f"Ошибка при получении списка моделей: {e}")
        return []

def model_response_generator(retriever, model, config):
    """Сгенерировать ответ с использованием модели и ретривера."""
    try:
        # logger.info("Генерация ответа с использованием модели и ретривера.")
        user_input = st.session_state["messages"][-1]["content"]

        # История сообщений
        if int(config['history_size']) and len(st.session_state["messages"]) > 0:
            message_list = [message['content'] for message in st.session_state["messages"]][:-1]
            last_messages = []
            for k, message in enumerate(message_list[::-1]):
                if k == int(config['history_size']):
                    break
                if k % 2:
                    last_messages.append(f'Q: {message}\n')
                else:
                    last_messages.append(f'A: {message}\n')
            message_history = f'\n\nИстория сообщений:\n{"".join(last_messages[::-1])}'
        else:
            message_history = ''

        # Обработка запросов
        maps_res = []
        if config['System_type'] in ['RAG', 'File']:
            shopping_res = search_shopping(user_input)
            internet_res, links, coordinates = search_places(user_input)
            maps_res = search_map(user_input, coordinates)  # Предполагаем, что это список строк

            # Формирование системного промпта
            system_prompt = (
        "Вы — цифровой помощник сервиса ВТБ Консьерж, нацеленный на предоставление точной, творчески оформленной информации. "
        "Ваши ответы должны быть не только полезными, но и эстетически приятными: лаконичными, логически структурированными, с элементами вдохновения и профессионализма. "
        "Соблюдайте рекомендации AMA: проявляйте уважение, позитив и эмпатию. Фокусируйтесь на интересах клиента, используя 'вы-подход'. "
        "Избегайте негативных формулировок и создавайте атмосферу уверенности и поддержки в каждом сообщении. "
        "Каждый ответ должен быть продуманным, элегантным и содержать одну законченную мысль, обогащенную конкретикой и практической ценностью. "
        "Если ответа на вопрос нет в контексте, честно сообщите об этом, но предложите альтернативные пути решения или дополнительные варианты, если это возможно. "
        "Используйте историю сообщений для построения ответа, если она предоставлена: \n\n{context}\n\n"
        "Если информация доступна из интернета, включайте её с корректными ссылками, формируя краткие и визуально понятные списки, например: " + str(internet_res) + " " + str(links) + "\n\n"
        "Когда уместно, предлагайте дополнительные объявления и рекомендации из данных: " + str(shopping_res) + "\n\n"
        "Информацию с Google Maps представляйте в виде удобной и читаемой таблицы: " + str(maps_res) + "\n\n"
        "Используйте формат ответа, который совмещает текст и таблицы. Например:\n"
        "- Начните с краткого и вдохновляющего введения.\n"
        "- Представьте информацию с использованием таблиц (если возможно) для удобного восприятия.\n"
        "- Завершите позитивным советом или рекомендацией.\n\n"
        "Пример структуры ответа:\n"
        "1. Введение: краткое, ясное и вдохновляющее.\n"
        "2. Таблицы для данных (например, для цен, рейтингов, мест и другой информации):\n"
        "```\n"
        "| Параметр          | Значение                 |\n"
        "|-------------------|--------------------------|\n"
        "| Средняя стоимость | 335,997 рублей          |\n"
        "| Рейтинг           | Высокий                 |\n"
        "```\n"
        "3. Заключение: ясный и позитивный вывод, например, предложение обратиться за дополнительной помощью или советы по выбору.\n\n"
        "Готовы приступить? Начинайте ответ в стильной, вдохновляющей манере!")


            if config['Model'].startswith('gpt'):
                prompt = ChatPromptTemplate.from_messages(
                    [("system", system_prompt), ("human", "{input}")]
                )
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
        logger.info("История сообщений инициализирована.")


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
        logger.info(f"Векторное хранилище инициализировано с {len(docs)} документами.")
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
        logger.info("Векторное хранилище для данных JSON инициализировано.")
        return vector_store

    else:
        logger.error(f"Неподдерживаемый тип разбиения: {splitter_type}")
        raise ValueError(f"Неподдерживаемый тип разбиения: {splitter_type}")

def create_vector_space(config):
    """Создать векторное пространство для извлечения документов на основе конфигурации."""
    logger.info(
        f"Инициализация векторного хранилища с моделью {config['Embedding']} и типом разбиения "
        f"{config['Splitter']['Type']}"
    )
    embeddings_model = config['Embedding']
    splitter_type = config['Splitter']['Type']
    chunk_size = int(config['Splitter']['Chunk_size'])

    
    documents = creating_documents(config)

    vector_store = create_retriever(splitter_type, embeddings_model, documents, chunk_size)
    
    return vector_store
