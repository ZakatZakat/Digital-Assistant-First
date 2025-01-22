from datasets import Dataset 
from ragas.testset.docstore import RunConfig
from langchain_core.language_models.llms import BaseLLM
from ragas import evaluate
from ragas.metrics import answer_correctness, answer_relevancy, context_precision, context_recall
import json
from langchain_core.vectorstores import VectorStore
from tqdm import tqdm
from utils.paths import CONTENT_DIR, ROOT_DIR, SRC_DIR
from utils.load_utils import load_model, load_vector_store, load_embeddings
from langchain_openai import ChatOpenAI
import yaml


def load_ragas_dataset(llm: BaseLLM, vector_store: VectorStore, qa_path: str, k: int):
    with open(qa_path, 'r') as rfile:
        data: dict = json.load(rfile)

    result = {'question': [], 'answer': [], 'ground_truth': [], 'contexts': []}
    print('Получение выводов от LLM...')
    for question, ground_truth in tqdm(data.items(), total=len(data)):
        prompt = """
        Представь, что ты менеджер банка ВТБ, который старается дать наиболее точный ответ на вопрос клиента. 
        Не выдумывай ничего, отвечай только на основании информации из контекста.

        Контекст: {context}

        Вопрос: {question}
        """

        # Получение контекста из векторного хранилища
        contexts = [ doc.page_content for doc in vector_store.similarity_search(query=question, k=k) ]
        context = '\n'.join(contexts)
        answer = llm.invoke(prompt.format(context=context, question=question))
        
        result['answer'].append(answer)
        result['ground_truth'].append(ground_truth)
        result['question'].append(question)
        result['contexts'].append(contexts)

    return Dataset.from_dict(result)


with open(ROOT_DIR / 'ragas_config.yaml', 'r') as rfile:
    config = yaml.safe_load(rfile)

# Загрузка модели-генератора
generator_llm = load_model(config['model'])
# Загрузка эмбеддингов
embeddings = load_embeddings(config['embeddings'])
# Загрузка векторного хранилища
vector_store = load_vector_store(
    config['vector_store'], 
    embeddings=embeddings, 
    path_to_context=CONTENT_DIR / 'global_js.json',
)

# Создание набора данных Ragas
dataset = load_ragas_dataset(
    generator_llm,
    vector_store=vector_store,
    qa_path=CONTENT_DIR / 'questions_0.json',
    k=config['k'],
)

print('Оценка ответа...')
# Оценка результатов
score = evaluate(
    dataset, 
    llm=generator_llm,
    embeddings=embeddings,
    metrics=[answer_correctness, ],
    run_config=RunConfig(timeout=900, max_retries=0, max_workers=1),
    raise_exceptions=True,
    # max_workers=1 для Ollama ускоряет вычисления
)
print(score.to_pandas())
score.to_pandas().to_csv(SRC_DIR / 'ragas_result.csv')