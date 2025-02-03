from typing import Dict, List

from src.offergen import Offer, offers_db, db_service
from src.offergen.agent import offer_matching_agent, PromptValidation, RagDeps
from src.offergen.vector_db import VectorDBService, Context, SearchResponse
from src.utils.logging import setup_logging
from src.utils.paths import ROOT_DIR as root_dir

logger = setup_logging(logging_path=str(root_dir / "logs" / "digital_assistant.log"))


def load_rag_examples(
    offers_db: Dict[int, Offer], query: str, db_service: VectorDBService
) -> List[dict]:
    """Load and filter RAG examples relevant to the query"""
    docs_and_scores = db_service.search(query, k=20)
    rag_data = SearchResponse(
        documents=[
            Context(content=doc.page_content, metadata=doc.metadata)
            for doc, _ in docs_and_scores
        ],
        scores=[score for _, score in docs_and_scores],
    )
    offers, scores, offer_ids = list[Offer](), list[float](), list[int]()
    for i, doc in enumerate(rag_data.documents):
        offer_id = int(doc.metadata["offer_url"].split("/")[-1])
        if offer_id in offers_db.keys() and offer_id not in offer_ids:
            offers.append(offers_db[offer_id])
            scores.append(rag_data.scores[i])
            offer_ids.append(offer_id)
    return offers, scores, offer_ids


def get_system_prompt_for_offers(
    validation_result: PromptValidation, prompt: str
) -> str:
    if not validation_result.is_valid:
        raise ValueError(
            f"Unable to generate system prompt for prompt: {prompt}. "
            f"Reason: {validation_result.reason}. "
            "Please ensure the request meets validation requirements."
        )

    logger.info(
        f"Loading RAG examples for prompt: {validation_result.modified_prompt_for_rag_search}"
    )
    offers, scores, offer_ids = load_rag_examples(
        offers_db, validation_result.modified_prompt_for_rag_search, db_service
    )
    logger.info(
        f"RAG examples loaded for prompt: {validation_result.modified_prompt_for_rag_search}"
    )

    # Create enhanced prompt with RAG context
    rag_context = "\nRelevant offer examples:\n"
    for offer, score, offer_id in zip(offers, scores, offer_ids):
        rag_context += f"- Category: {offer.category}\n"
        rag_context += f"- Title: {offer.name}\n"
        rag_context += f"- Short description: {offer.short_description}\n"
        rag_context += f"- Full description: {offer.full_description}\n"
        rag_context += f"- Offer RAG score: {score}\n"
        rag_context += f"- Offer ID: {offer_id}\n"
        rag_context += "---\n"

    enhanced_prompt = f"{rag_context}\nUser request: {prompt}"

    # Get offer matches
    deps = RagDeps(k=validation_result.number_of_offers_to_generate, offers=offers_db)
    result = offer_matching_agent.run_sync(enhanced_prompt, deps=deps)
    logger.info(f"Offer matching agent result: {result.data}")

    # Stage 3: Format the output
    if result and result.data.matches:
        information_about_relevant_offers = str()
        for match in result.data.matches:
            offer = offers_db[match.offer_id]
            information_about_relevant_offers += f"Offer ID: {match.offer_id}\n"
            information_about_relevant_offers += f"Offer name: {offer.name}\n"
            information_about_relevant_offers += f"Offer category: {offer.category}\n"
            information_about_relevant_offers += (
                f"Offer short description: {offer.short_description}\n"
            )
            information_about_relevant_offers += (
                f"Offer full description: {offer.full_description}\n"
            )
            information_about_relevant_offers += f"Offer URL: {offer.offer_url}\n"
            information_about_relevant_offers += (
                f"Offer match reason: {match.match_reason}\n"
            )
            information_about_relevant_offers += "---\n"
        logger.info("System prompt for offers generated.")
        return f"""
You are a VTB Family offers formatter. Format and evaluate these offers:

{information_about_relevant_offers}

The input you receive is the user's initial search request. Use it to evaluate offer relevance.

Main tasks:
1. Format search results in markdown
2. Structure each offer clearly
3. Match offers against the initial request
4. Write everything in Russian

Write a summary that:
- Mentions the initial search request
- States how well the offers match the request
- Explains any mismatches and their potential value
- Speaks directly to the user who made the request

Use this format for each offer:
### [OFFER TITLE]
**Категория:** [CATEGORY]

**Описание предложения:**
[SHORT, CONCISE DESCRIPTION OF THE MAIN OFFER/DISCOUNT]

**Информация о компании:**
- Адрес: [ADDRESS IF AVAILABLE]
- Телефон: [PHONE IF AVAILABLE]
- Часы работы: [HOURS IF AVAILABLE]
- Сайт: [WEBSITE IF AVAILABLE]

**Ссылка на предложение:** [Подробнее на VTB Family]([OFFER URL])

---

Key requirements:
- Pull company details from the full description
- Keep descriptions brief and value-focused
- Connect your response to the user's search request
"""
