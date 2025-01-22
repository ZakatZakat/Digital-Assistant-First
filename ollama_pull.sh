#!/bin/bash

# Wait for ollama serve to be ready
echo "Waiting for ollama serve to be ready..."
until nc -z localhost 11434; do
    sleep 1
done

# Run ollama pull command
ollama pull llama3.2:latest