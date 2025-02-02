#Импорты стандартной библиотеки
import logging

from serpapi import GoogleSearch
from src.utils.check_serp_response import APIKeyManager

# Локальные импорты
from src.utils.paths import ROOT_DIR

# Настройка логирования
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

serpapi_key_manager = APIKeyManager(path_to_file="api_keys_status.xlsx")


def search_map(q, coordinates):
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

    return good_results


def search_shopping(q):
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
    return results_with_titles_and_links

def search_places(q):
    """Search for places using Google Search API, возвращает только первые 5 результатов."""
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

    return good_results, results_with_titles_and_links, coordinates
