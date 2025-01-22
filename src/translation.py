from transformers import MarianMTModel, MarianTokenizer
import json

# Загрузка модели и токенизатора
model_name = 'Helsinki-NLP/opus-mt-ru-en'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

def translate_text(text):
    """Функция для перевода текста с русского на английский."""
    translated = model.generate(**tokenizer(text, return_tensors="pt", padding=True))
    return tokenizer.decode(translated[0], skip_special_tokens=True)

def translate_json(json_data):
    """Рекурсивная функция для перевода всех ключей и значений в JSON объекте."""
    if isinstance(json_data, dict):
        translated_dict = {}
        for key, value in json_data.items():
            translated_key = translate_text(key)
            translated_value = translate_json(value)
            translated_dict[translated_key] = translated_value
        return translated_dict
    elif isinstance(json_data, list):
        return [translate_json(item) for item in json_data]
    elif isinstance(json_data, str):
        return translate_text(json_data)
    else:
        return json_data

# Загрузка вашего JSON файла
with open('content/global_js.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Перевод всех ключей и значений в JSON
translated_data = translate_json(data)

# Сохранение переведенного JSON файла
with open('content/translated.json', 'w', encoding='utf-8') as f:
    json.dump(translated_data, f, ensure_ascii=False, indent=4)