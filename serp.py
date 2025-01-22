import requests

serpapi_key = "eccab63b4cc91c7e9a8dfd2c6a5a0e17ad40beb185de293570050664e3f60b2b"

def get_serpapi_usage(api_key):
    url = f"https://serpapi.com/account?api_key={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        # Пример вывода нужных полей:
        print(f"Ваша почта: {data.get('account_email')}")
        print(f"Тарифный план: {data.get('billing')}")
        print(f"Всего запросов по плану: {data.get('requests_allotted')}")
        print(f"Использовано запросов: {data.get('requests_used')}")
        print(f"Осталось запросов: {data.get('requests_remaining')}")
    else:
        print(f"Ошибка {response.status_code}: {response.text}")

# Пример вызова
get_serpapi_usage(serpapi_key)
