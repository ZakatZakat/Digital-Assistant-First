import yaml
import subprocess
from utils.paths import ROOT_DIR


def install_ollama():
    """Установить Ollama с использованием предоставленного скрипта установки."""
    subprocess.run(['curl', '-fsSL', 'https://ollama.com/install.sh', '|', 'sh'], check=True, shell=True)

def pull_model(model_name: str):
    """Загрузить определённую модель из Ollama."""
    print(f"Загрузка {model_name}...")
    subprocess.run(['ollama', 'pull', model_name], check=True)

def main():
    # Загрузка конфигурации
    with open(ROOT_DIR / 'config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Получение имени модели из конфигурации
    model = config.get('Model')

    if model:
        install_ollama()

        if model == 'All':
            models_to_process = ['llama3.1:8b-instruct-q8_0', 'llama3.1:latest', 'gemma2:9b', 'mistral-nemo:12b']
            for model_name in models_to_process:
                pull_model(model_name)
        else:
            pull_model(model)
    
    print("Все модели были загружены.")

if __name__ == "__main__":
    main()