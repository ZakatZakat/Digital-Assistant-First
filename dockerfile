# Использовать базовый образ
FROM ubuntu:22.04

WORKDIR /app

# Обновление и установка зависимостей
RUN apt-get update -y && \
    apt-get install -y build-essential python3-pip python3-dev curl bash supervisor netcat && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Установка Ollama
RUN for i in {1..5}; do \
    curl -sSfL https://ollama.ai/install.sh | bash && break || \
    echo "Повтор попытки... ($i/5)" && sleep 5; \
    done

# Добавить Ollama в PATH
ENV PATH="/root/.ollama/bin:${PATH}"

# Копировать код приложения
COPY . /app

# Установка зависимостей Python
RUN for i in {1..5}; do \
    pip install -r requirements.txt && break || \
    echo "Повтор попытки... ($i/5)" && sleep 5; \
    done

# Копировать конфигурацию supervisord
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

ENV PYTHONPATH=".:./src"

# Копировать скрипт ollama_pull.sh
COPY ollama_pull.sh /app/ollama_pull.sh
RUN chmod +x /app/ollama_pull.sh

# Открыть необходимые порты
EXPOSE 8080

# Запустить Supervisor
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]