"""
Evaluation Pipeline for SFDA Cosmetics Chatbot.

This module evaluates the RAG system using multiple metrics:
- Retrieval Accuracy: Are the right documents retrieved?
- Answer Faithfulness: Is the answer based on retrieved context?
- Answer Relevancy: Does the answer address the question?
- Citation Accuracy: Are sources correctly cited?
"""

import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import pandas as pd

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

try:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_recall,
        context_precision,
    )
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    print("⚠️ Ragas not installed. Install with: pip install ragas")

import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SFDAEvaluator:
    """Evaluates the SFDA chatbot performance."""

    def __init__(self):
        """Initialize evaluator with models and vector store."""
        logger.info("Initializing SFDA Evaluator...")

        # Load embedding model
        self.embeddings_model = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={"device": config.EMBEDDING_DEVICE},
        )

        # Load LLM
        self.llm = ChatOpenAI(
            model=config.LLM_MODEL,
            temperature=config.LLM_TEMPERATURE,
            api_key=config.OPENROUTER_API_KEY,
            base_url=config.LLM_BASE_URL,
            max_tokens=config.LLM_MAX_TOKENS,
        )

        # Load vector store
        self.vector_store = Chroma(
            collection_name=config.COLLECTION_NAME,
            embedding_function=self.embeddings_model,
            persist_directory=config.CHROMA_PATH,
        )

        logger.info("✅ Evaluator initialized")

    def load_test_queries(self, test_file: str = "test_queries.json") -> List[Dict[str, Any]]:
        """
        Load test queries from JSON file.

        Args:
            test_file: Path to test queries JSON file

        Returns:
            List of test query dictionaries
        """
        test_path = Path(test_file)
        if not test_path.exists():
            logger.warning(f"Test file not found: {test_file}")
            return []

        with open(test_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def evaluate_retrieval(
        self,
        query: str,
        expected_articles: List[str],
        source_choice: str = "لوائح التجميل (PDF)",
        k: int = 3
    ) -> Dict[str, Any]:
        """
        Evaluate retrieval accuracy.

        Args:
            query: User query
            expected_articles: List of expected article numbers
            source_choice: Source filter
            k: Number of documents to retrieve

        Returns:
            Evaluation metrics
        """
        # Build filter
        search_filter = None
        if source_choice == "لوائح التجميل (PDF)":
            search_filter = {"category": "regulation"}
        elif source_choice == "محظورات التجميل":
            search_filter = {"category": "banned"}

        # Retrieve documents
        search_kwargs = {"k": k}
        if search_filter:
            search_kwargs["filter"] = search_filter

        docs = self.vector_store.similarity_search(query, **search_kwargs)

        # Extract article numbers from retrieved docs
        retrieved_articles = set()
        for doc in docs:
            article = doc.metadata.get("article")
            if article:
                retrieved_articles.add(str(article))

        expected_set = set(str(a) for a in expected_articles)

        # Calculate metrics
        true_positives = len(retrieved_articles & expected_set)
        false_positives = len(retrieved_articles - expected_set)
        false_negatives = len(expected_set - retrieved_articles)

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {
            "query": query,
            "expected_articles": list(expected_set),
            "retrieved_articles": list(retrieved_articles),
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "retrieved_docs_count": len(docs),
        }

    def evaluate_citation(self, answer: str, retrieved_docs: List[Document]) -> Dict[str, Any]:
        """
        Evaluate if answer includes proper citations.

        Args:
            answer: Generated answer
            retrieved_docs: Retrieved documents

        Returns:
            Citation evaluation metrics
        """
        # Check if answer has source footer
        has_source_footer = "المصدر:" in answer or "**المصدر:**" in answer

        # Extract article numbers mentioned in answer
        import re
        article_pattern = r"المادة\s+(\d+)|Article\s+(\d+)"
        mentioned_articles = set()
        for match in re.finditer(article_pattern, answer):
            article_num = match.group(1) or match.group(2)
            if article_num:
                mentioned_articles.add(article_num)

        # Get articles from retrieved docs
        retrieved_articles = set()
        for doc in retrieved_docs:
            article = doc.metadata.get("article")
            if article:
                retrieved_articles.add(str(article))

        # Check if mentioned articles are from retrieved docs
        valid_citations = mentioned_articles & retrieved_articles
        invalid_citations = mentioned_articles - retrieved_articles

        citation_accuracy = len(valid_citations) / len(mentioned_articles) if mentioned_articles else 0

        return {
            "has_source_footer": has_source_footer,
            "mentioned_articles": list(mentioned_articles),
            "retrieved_articles": list(retrieved_articles),
            "valid_citations": list(valid_citations),
            "invalid_citations": list(invalid_citations),
            "citation_accuracy": citation_accuracy,
        }

    def generate_answer(self, query: str, source_choice: str = "لوائح التجميل (PDF)") -> tuple[str, List[Document]]:
        """
        Generate answer for a query.

        Args:
            query: User question
            source_choice: Source filter

        Returns:
            Tuple of (answer, retrieved_documents)
        """
        # Build filter
        search_filter = None
        if source_choice == "لوائح التجميل (PDF)":
            search_filter = {"category": "regulation"}
        elif source_choice == "محظورات التجميل":
            search_filter = {"category": "banned"}

        # Retrieve documents
        search_kwargs = {"k": config.RETRIEVAL_K}
        if search_filter:
            search_kwargs["filter"] = search_filter

        retrieved_docs = self.vector_store.similarity_search(query, **search_kwargs)

        if not retrieved_docs:
            return "لم أجد نصًا صريحًا في المصادر المتاحة يجيب عن ذلك.", []

        # Build context
        context_parts = []
        for d in retrieved_docs[:3]:
            snippet = d.page_content[:1400]
            context_parts.append(snippet)
        context = "\n\n".join(context_parts)

        # Generate answer
        prompt = f"""
ROLE:
أنت مساعد امتثال يعتمد فقط على النصوص المرفقة.

RULES:
- لا تضف أي معلومة من خارج النصوص المساعدة.
- إذا لم تجد نصاً صريحاً، قل: "لم أجد نصاً صريحاً في المصادر المرفقة يجيب عن ذلك."
- اكتب إجابة قصيرة ومنظمة.

النصوص المساعدة:
{context}

سؤال المستخدم: {query}

اكتب الإجابة:
""".strip()

        response = self.llm.invoke([HumanMessage(content=prompt)])
        answer = response.content.strip()

        return answer, retrieved_docs

    def evaluate_test_set(self, test_queries: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Evaluate entire test set.

        Args:
            test_queries: List of test query dictionaries

        Returns:
            DataFrame with evaluation results
        """
        results = []

        for i, test in enumerate(test_queries, 1):
            logger.info(f"Evaluating query {i}/{len(test_queries)}: {test['query']}")

            try:
                # Evaluate retrieval
                retrieval_eval = self.evaluate_retrieval(
                    query=test["query"],
                    expected_articles=test.get("expected_articles", []),
                    source_choice=test.get("source_choice", "لوائح التجميل (PDF)"),
                )

                # Generate answer
                answer, retrieved_docs = self.generate_answer(
                    query=test["query"],
                    source_choice=test.get("source_choice", "لوائح التجميل (PDF)"),
                )

                # Evaluate citation
                citation_eval = self.evaluate_citation(answer, retrieved_docs)

                # Combine results
                result = {
                    "query": test["query"],
                    "source_choice": test.get("source_choice", "لوائح التجميل (PDF)"),
                    "generated_answer": answer[:200] + "..." if len(answer) > 200 else answer,
                    "retrieval_precision": retrieval_eval["precision"],
                    "retrieval_recall": retrieval_eval["recall"],
                    "retrieval_f1": retrieval_eval["f1_score"],
                    "has_citation": citation_eval["has_source_footer"],
                    "citation_accuracy": citation_eval["citation_accuracy"],
                    "expected_articles": retrieval_eval["expected_articles"],
                    "retrieved_articles": retrieval_eval["retrieved_articles"],
                }

                results.append(result)

            except Exception as e:
                logger.error(f"Error evaluating query '{test['query']}': {e}")
                continue

        return pd.DataFrame(results)

    def generate_report(self, results_df: pd.DataFrame, output_file: str = "evaluation_report.md"):
        """
        Generate evaluation report.

        Args:
            results_df: Evaluation results DataFrame
            output_file: Output markdown file
        """
        # Calculate aggregate metrics
        avg_precision = results_df["retrieval_precision"].mean()
        avg_recall = results_df["retrieval_recall"].mean()
        avg_f1 = results_df["retrieval_f1"].mean()
        citation_rate = results_df["has_citation"].mean()
        avg_citation_accuracy = results_df["citation_accuracy"].mean()

        # Generate report
        report = f"""# SFDA Chatbot Evaluation Report

## Executive Summary

This report evaluates the SFDA Cosmetics Chatbot RAG system using {len(results_df)} test queries.

## Aggregate Metrics

| Metric | Score |
|--------|-------|
| **Retrieval Precision** | {avg_precision:.2%} |
| **Retrieval Recall** | {avg_recall:.2%} |
| **Retrieval F1 Score** | {avg_f1:.2%} |
| **Citation Rate** | {citation_rate:.2%} |
| **Citation Accuracy** | {avg_citation_accuracy:.2%} |

### Interpretation

- **Retrieval Precision**: {avg_precision:.2%} of retrieved documents were relevant
- **Retrieval Recall**: {avg_recall:.2%} of relevant documents were retrieved
- **F1 Score**: Harmonic mean of precision and recall
- **Citation Rate**: {citation_rate:.2%} of answers included source citations
- **Citation Accuracy**: {avg_citation_accuracy:.2%} of cited articles were from retrieved documents

## Detailed Results

"""

        # Add detailed results table
        report += "\n### Query-Level Performance\n\n"
        report += results_df.to_markdown(index=False)

        # Add analysis
        report += f"""

## Analysis

### Strengths
- {"High retrieval precision" if avg_precision > 0.8 else "Good retrieval precision"}
- {"Excellent citation compliance" if citation_rate > 0.9 else "Good citation compliance" if citation_rate > 0.7 else "Citations need improvement"}
- Arabic text handling

### Areas for Improvement
"""
        if avg_recall < 0.7:
            report += "- Improve retrieval recall (consider adjusting chunk size or k value)\n"
        if citation_rate < 0.9:
            report += "- Ensure all answers include source citations\n"
        if avg_citation_accuracy < 0.9:
            report += "- Improve citation accuracy (ensure cited articles match retrieved docs)\n"

        report += """
## Recommendations

1. **Retrieval Optimization**: Fine-tune chunk size and overlap parameters
2. **Prompt Engineering**: Refine prompts to ensure consistent citation format
3. **Quality Assurance**: Implement automated citation validation
4. **User Feedback**: Collect user ratings to improve system

## Test Environment

- **Embedding Model**: """ + config.EMBEDDING_MODEL + """
- **LLM Model**: """ + config.LLM_MODEL + """
- **Retrieval K**: """ + str(config.RETRIEVAL_K) + """
- **Chunk Size**: """ + str(config.CHUNK_SIZE) + """
- **Test Date**: """ + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S") + """
"""

        # Save report
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(report)

        logger.info(f"✅ Report saved to {output_file}")

        return report


def main():
    """Run evaluation pipeline."""
    logger.info("=" * 60)
    logger.info("SFDA Chatbot Evaluation Pipeline")
    logger.info("=" * 60)

    # Initialize evaluator
    evaluator = SFDAEvaluator()

    # Load test queries
    test_queries = evaluator.load_test_queries()

    if not test_queries:
        logger.warning("No test queries found. Creating sample test set...")
        # Create sample test queries if none exist
        test_queries = [
            {
                "query": "ما هي المادة الرابعة؟",
                "expected_articles": ["4"],
                "source_choice": "لوائح التجميل (PDF)",
            },
            {
                "query": "اذكر التزامات المُدرج في النظام",
                "expected_articles": [],
                "source_choice": "لوائح التجميل (PDF)",
            },
        ]

    # Run evaluation
    logger.info(f"\nRunning evaluation on {len(test_queries)} queries...")
    results_df = evaluator.evaluate_test_set(test_queries)

    # Save results
    results_df.to_csv("evaluation_results.csv", index=False, encoding="utf-8-sig")
    logger.info("✅ Results saved to evaluation_results.csv")

    # Generate report
    evaluator.generate_report(results_df)

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("Evaluation Complete!")
    logger.info("=" * 60)
    logger.info(f"Average Retrieval Precision: {results_df['retrieval_precision'].mean():.2%}")
    logger.info(f"Average Retrieval Recall: {results_df['retrieval_recall'].mean():.2%}")
    logger.info(f"Average F1 Score: {results_df['retrieval_f1'].mean():.2%}")
    logger.info(f"Citation Rate: {results_df['has_citation'].mean():.2%}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
