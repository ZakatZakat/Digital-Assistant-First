import logging
from datetime import datetime
from pathlib import Path
import json

def setup_logging(logging_path='logs/digital_assistant.log'):
    """Настройка логирования"""
    # Создаем директорию для логов, если она не существует
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    # Настраиваем формат логов
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler(logging_path, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def log_api_call(logger: logging.Logger, source: str, request: str, response: str, error: str = None):
    """
    Логирование API вызовов
    
    Args:
        source: Источник запроса (LLM/SerpAPI)
        request: Текст запроса
        response: Текст ответа
        error: Сообщение об ошибке (если есть)
    """
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'source': source,
        'request': request,
        'response': response,
    }
    if error:
        log_entry['error'] = error
    
    logger.info(f"API Call: {json.dumps(log_entry, ensure_ascii=False)}")
    