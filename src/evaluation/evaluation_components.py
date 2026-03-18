"""
Evaluation Components for Evaluation & Tracing Pipeline (Phase 11)

Implements core components for RAG evaluation including RAGAS metrics,
LangSmith tracing, regression testing, and comprehensive reporting.
"""

import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data_ingestion.config import Config
from src.data_ingestion.logging_config import create_logger

logger = create_logger(__name__)

# Try importing RAGAS and LangSmith
try:
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    RAGAS_AVAILABLE = True
except ImportError:
    logger.warning("RAGAS not available - install with: pip install ragas")
    RAGAS_AVAILABLE = False

try:
    from langchain_core.callbacks import LangChainTracer
    from langsmith import Client as LangSmithClient
    from langsmith.wrappers import wrap_openai

    LANGSMITH_AVAILABLE = True
except ImportError:
    logger.warning("LangSmith not available - install with: pip install langsmith")
    LANGSMITH_AVAILABLE = False


class RAGEvaluator:
    """
    Evaluates RAG system performance on custom test queries.

    Measures retrieval latency, generation latency, and response quality
    by running test queries through the full RAG pipeline.
    """

    def __init__(
        self,
        llm_client,
        retrieval_pipeline=None,
        generation_pipeline=None,
        timeout_seconds: float = 30.0,
    ):
        """
        Initialize RAG evaluator.

        Args:
            llm_client: LLM client for evaluation queries
            retrieval_pipeline: Retrieval pipeline instance
            generation_pipeline: Generation pipeline instance
            timeout_seconds: Timeout for each query evaluation
        """
        self.llm_client = llm_client
        self.retrieval_pipeline = retrieval_pipeline
        self.generation_pipeline = generation_pipeline
        self.timeout_seconds = timeout_seconds

        logger.info(
            f"RAGEvaluator initialized",
            extra_data={
                "has_retrieval": retrieval_pipeline is not None,
                "has_generation": generation_pipeline is not None,
                "timeout": timeout_seconds,
            },
        )

    def evaluate_single_query(self, query: str, verbose: bool = False) -> Dict:
        """
        Evaluate single query through full RAG pipeline.

        Args:
            query: Query to evaluate
            verbose: Whether to log details

        Returns:
            Dict with latencies, response, and metadata
        """
        start_time = time.time()
        result = {
            "query": query,
            "latencies": {"retrieval_ms": 0, "generation_ms": 0, "total_ms": 0},
            "retrieved_docs": [],
            "response": "",
            "error": None,
        }

        try:
            # Phase 1: Retrieval
            if self.retrieval_pipeline:
                retrieval_start = time.time()
                try:
                    retrieval_result = self.retrieval_pipeline.run_complete_pipeline(
                        query, verbose=False
                    )
                    result["latencies"]["retrieval_ms"] = (
                        time.time() - retrieval_start
                    ) * 1000
                    result["retrieved_docs"] = retrieval_result.get(
                        "final_documents", []
                    )

                    if verbose:
                        logger.debug(
                            f"Retrieval completed",
                            extra_data={
                                "latency_ms": result["latencies"]["retrieval_ms"],
                                "docs_retrieved": len(result["retrieved_docs"]),
                            },
                        )
                except Exception as e:
                    logger.error(f"Retrieval failed: {str(e)}")
                    result["error"] = f"Retrieval error: {str(e)}"
                    result["latencies"]["total_ms"] = (time.time() - start_time) * 1000
                    return result

            # Phase 2: Generation
            if self.generation_pipeline:
                generation_start = time.time()
                try:
                    generation_result = self.generation_pipeline.generate_response(
                        query, result["retrieved_docs"], verbose=False
                    )
                    result["latencies"]["generation_ms"] = (
                        time.time() - generation_start
                    ) * 1000
                    result["response"] = generation_result.get("response", "")

                    if verbose:
                        logger.debug(
                            f"Generation completed",
                            extra_data={
                                "latency_ms": result["latencies"]["generation_ms"],
                                "response_length": len(result["response"]),
                            },
                        )
                except Exception as e:
                    logger.error(f"Generation failed: {str(e)}")
                    result["error"] = f"Generation error: {str(e)}"
                    result["latencies"]["total_ms"] = (time.time() - start_time) * 1000
                    return result

            result["latencies"]["total_ms"] = (time.time() - start_time) * 1000

            logger.debug(
                f"Query evaluation completed",
                extra_data={"total_latency_ms": result["latencies"]["total_ms"]},
            )

            return result

        except Exception as e:
            logger.error(f"Error evaluating query: {str(e)}")
            result["error"] = str(e)
            result["latencies"]["total_ms"] = (time.time() - start_time) * 1000
            return result

    def batch_evaluate(self, queries: List[str], verbose: bool = False) -> List[Dict]:
        """
        Evaluate batch of queries.

        Args:
            queries: List of queries to evaluate
            verbose: Whether to log details

        Returns:
            List of evaluation results
        """
        logger.info(
            f"Starting batch evaluation", extra_data={"query_count": len(queries)}
        )

        results = []

        for i, query in enumerate(queries, 1):
            try:
                if verbose:
                    logger.info(f"Evaluating query {i}/{len(queries)}: {query[:50]}...")

                result = self.evaluate_single_query(query, verbose=verbose)
                results.append(result)

            except Exception as e:
                logger.error(f"Error in batch evaluation for query {i}: {str(e)}")
                results.append(
                    {"query": query, "error": str(e), "latencies": {"total_ms": 0}}
                )

        logger.info(
            f"Batch evaluation completed",
            extra_data={
                "total_queries": len(queries),
                "successful": len([r for r in results if "error" not in r]),
                "failed": len([r for r in results if "error" in r]),
            },
        )

        return results

    def evaluate_answer_relevance(self, query: str, answer: str) -> Dict:
        """
        Evaluate relevance of answer to query.

        Args:
            query: User query
            answer: Generated answer

        Returns:
            Dict with relevance score
        """
        try:
            # Simple relevance scoring based on overlap
            query_words = set(query.lower().split())
            answer_words = set(answer.lower().split())

            if not query_words:
                return {"relevance_score": 0.0, "status": "error"}

            overlap = len(query_words & answer_words)
            relevance_score = min(overlap / len(query_words), 1.0)

            logger.info(
                f"Answer relevance evaluated",
                extra_data={
                    "query_length": len(query),
                    "answer_length": len(answer),
                    "relevance_score": relevance_score,
                },
            )

            return {"relevance_score": relevance_score, "status": "success"}
        except Exception as e:
            logger.error(f"Error evaluating answer relevance: {str(e)}")
            return {"relevance_score": 0.0, "status": "error", "error": str(e)}

    def evaluate_faithfulness(self, context: str, answer: str) -> Dict:
        """
        Evaluate whether answer is faithful to the provided context.

        Args:
            context: Reference context
            answer: Generated answer

        Returns:
            Dict with faithfulness score
        """
        try:
            # Simple faithfulness check: verify answer content is in context
            context_lower = context.lower()
            answer_lower = answer.lower()

            # Count matching phrases
            answer_words = answer_lower.split()
            matching_words = sum(1 for word in answer_words if word in context_lower)

            if not answer_words:
                return {"faithfulness_score": 0.0, "status": "error"}

            faithfulness_score = matching_words / len(answer_words)

            logger.info(
                f"Faithfulness evaluated",
                extra_data={
                    "context_length": len(context),
                    "answer_length": len(answer),
                    "faithfulness_score": faithfulness_score,
                },
            )

            return {"faithfulness_score": faithfulness_score, "status": "success"}
        except Exception as e:
            logger.error(f"Error evaluating faithfulness: {str(e)}")
            return {"faithfulness_score": 0.0, "status": "error", "error": str(e)}

    def evaluate_context_precision(self, query: str, context: List) -> Dict:
        """
        Evaluate precision of retrieved context.

        Args:
            query: User query
            context: List of retrieved Document objects

        Returns:
            Dict with precision score
        """
        try:
            if not context:
                return {"precision_score": 0.0, "status": "success"}

            # Simple precision: measure relevance of each document to query
            query_words = set(query.lower().split())
            relevant_docs = 0

            for doc in context:
                doc_text = getattr(doc, "page_content", str(doc)).lower()
                doc_words = set(doc_text.split())

                # Document is relevant if it shares >10% of query words
                if len(query_words & doc_words) / len(query_words) > 0.1:
                    relevant_docs += 1

            precision_score = relevant_docs / len(context) if context else 0.0

            logger.info(
                f"Context precision evaluated",
                extra_data={
                    "total_docs": len(context),
                    "relevant_docs": relevant_docs,
                    "precision_score": precision_score,
                },
            )

            return {"precision_score": precision_score, "status": "success"}
        except Exception as e:
            logger.error(f"Error evaluating context precision: {str(e)}")
            return {"precision_score": 0.0, "status": "error", "error": str(e)}

    def evaluate_context_recall(self, query: str, answer: str, context: List) -> Dict:
        """
        Evaluate recall: whether all relevant information is in context.

        Args:
            query: User query
            answer: Generated answer
            context: List of retrieved Document objects

        Returns:
            Dict with recall score
        """
        try:
            if not context:
                return {"recall_score": 0.0, "status": "success"}

            # Combine all context
            full_context = " ".join(
                getattr(doc, "page_content", str(doc)) for doc in context
            ).lower()
            answer_lower = answer.lower()

            # Count answer words present in context
            answer_words = answer_lower.split()
            found_words = sum(1 for word in answer_words if word in full_context)

            recall_score = found_words / len(answer_words) if answer_words else 0.0

            logger.info(
                f"Context recall evaluated",
                extra_data={
                    "total_docs": len(context),
                    "found_words": found_words,
                    "total_words": len(answer_words),
                    "recall_score": recall_score,
                },
            )

            return {"recall_score": recall_score, "status": "success"}
        except Exception as e:
            logger.error(f"Error evaluating context recall: {str(e)}")
            return {"recall_score": 0.0, "status": "error", "error": str(e)}


class RAGASMetricsEvaluator:
    """
    RAGAS framework for RAG evaluation metrics.

    Computes ground-truth-free metrics including faithfulness,
    answer relevancy, context precision, and context recall.
    """

    def __init__(self):
        """Initialize RAGAS evaluator."""
        if not RAGAS_AVAILABLE:
            logger.warning("RAGAS not available - metrics evaluation will be skipped")

        self.metrics = []
        if RAGAS_AVAILABLE:
            self.metrics = [
                faithfulness,  # Does answer come from context?
                answer_relevancy,  # Is answer relevant to query?
                context_precision,  # Are retrieved docs relevant?
                context_recall,  # Did we get all relevant docs?
            ]

        logger.info(
            f"RAGASMetricsEvaluator initialized",
            extra_data={"metrics_available": len(self.metrics)},
        )

    def prepare_evaluation_dataset(
        self, evaluation_results: List[Dict]
    ) -> Optional[object]:
        """
        Convert evaluation results to RAGAS Dataset format.

        Args:
            evaluation_results: List of evaluation results

        Returns:
            RAGAS Dataset or None if RAGAS unavailable
        """
        try:
            if not RAGAS_AVAILABLE:
                logger.warning("Cannot prepare dataset - RAGAS not available")
                return None

            data_dict = {
                "question": [],
                "answer": [],
                "contexts": [],
            }

            for result in evaluation_results:
                data_dict["question"].append(result.get("query", ""))
                data_dict["answer"].append(result.get("response", ""))

                # Extract context from retrieved docs
                contexts = []
                for doc in result.get("retrieved_docs", []):
                    content = doc.get("page_content", "") or doc.get("content", "")
                    if content:
                        contexts.append(content)

                data_dict["contexts"].append(contexts)

            dataset = Dataset.from_dict(data_dict)

            logger.info(
                f"Dataset prepared for RAGAS",
                extra_data={"samples": len(dataset), "columns": dataset.column_names},
            )

            return dataset
        except Exception as e:
            logger.error(f"Error preparing dataset: {str(e)}")
            return None

    def evaluate(self, evaluation_results: List[Dict]) -> Dict:
        """
        Evaluate RAG results using RAGAS metrics.

        Args:
            evaluation_results: List of evaluation results

        Returns:
            Dict with metric scores and analysis
        """
        try:
            if not RAGAS_AVAILABLE:
                logger.warning("RAGAS not available - returning empty metrics")
                return {
                    "error": "RAGAS not available",
                    "sample_count": len(evaluation_results),
                }

            logger.info(f"Starting RAGAS evaluation")

            dataset = self.prepare_evaluation_dataset(evaluation_results)
            if dataset is None:
                return {"error": "Failed to prepare dataset"}

            logger.debug(f"Evaluating with {len(self.metrics)} metrics")

            ragas_results = evaluate(
                dataset=dataset, metrics=self.metrics, raise_exceptions=False
            )

            results_dict = (
                ragas_results.to_dict()
                if hasattr(ragas_results, "to_dict")
                else ragas_results
            )

            overall_scores = {}
            for metric in self.metrics:
                metric_name = metric.name if hasattr(metric, "name") else str(metric)
                overall_scores[metric_name] = results_dict.get(metric_name, None)

            result = {
                "overall_scores": overall_scores,
                "individual_scores": results_dict,
                "sample_count": len(dataset),
            }

            logger.info(
                f"RAGAS evaluation completed",
                extra_data={"metrics_computed": len(overall_scores)},
            )

            return result

        except Exception as e:
            logger.error(f"Error during RAGAS evaluation: {str(e)}")
            return {
                "error": str(e),
                "individual_scores": None,
                "sample_count": len(evaluation_results),
            }

    def print_metric_report(self, metrics_results: Dict):
        """Print formatted metrics report."""
        try:
            logger.info("Printing metrics report")

            overall_scores = metrics_results.get("overall_scores", {})
            if overall_scores:
                logger.info("Overall Scores:")
                for metric_name, score in overall_scores.items():
                    if score is not None:
                        logger.info(f"  {metric_name}: {score:.4f}")
                    else:
                        logger.info(f"  {metric_name}: N/A")

            if "sample_count" in metrics_results:
                logger.info(f"Samples Evaluated: {metrics_results['sample_count']}")

            if "error" in metrics_results:
                logger.warning(f"Evaluation Error: {metrics_results['error']}")

        except Exception as e:
            logger.error(f"Error printing metrics report: {str(e)}")

    def compute_faithfulness(self, question: str, context: str, answer: str) -> float:
        """
        Compute faithfulness score - is answer faithful to context?

        Args:
            question: User question
            context: Reference context
            answer: Generated answer

        Returns:
            Faithfulness score (0-1)
        """
        try:
            context_lower = context.lower()
            answer_lower = answer.lower()

            # Count matching words
            answer_words = answer_lower.split()
            matching_words = sum(1 for word in answer_words if word in context_lower)

            score = matching_words / len(answer_words) if answer_words else 0.0

            logger.info(
                f"Faithfulness computed",
                extra_data={"score": score, "matching_words": matching_words},
            )

            return min(score, 1.0)
        except Exception as e:
            logger.error(f"Error computing faithfulness: {str(e)}")
            return 0.0

    def compute_answer_relevance(self, question: str, answer: str) -> float:
        """
        Compute answer relevance score - is answer relevant to question?

        Args:
            question: User question
            answer: Generated answer

        Returns:
            Relevance score (0-1)
        """
        try:
            question_words = set(question.lower().split())
            answer_words = set(answer.lower().split())

            if not question_words:
                return 0.0

            overlap = len(question_words & answer_words)
            score = overlap / len(question_words)

            logger.info(
                f"Answer relevance computed",
                extra_data={"score": score, "overlap": overlap},
            )

            return min(score, 1.0)
        except Exception as e:
            logger.error(f"Error computing answer relevance: {str(e)}")
            return 0.0

    def compute_context_precision(self, question: str, context: List) -> float:
        """
        Compute context precision - are retrieved docs relevant to question?

        Args:
            question: User question
            context: List of context documents

        Returns:
            Precision score (0-1)
        """
        try:
            if not context:
                return 0.0

            question_words = set(question.lower().split())
            relevant_docs = 0

            for doc in context:
                doc_text = (
                    getattr(doc, "page_content", str(doc))
                    if hasattr(doc, "page_content")
                    else str(doc)
                )
                doc_words = set(doc_text.lower().split())

                # Document is relevant if it shares >10% of question words
                if len(question_words & doc_words) / len(question_words) > 0.1:
                    relevant_docs += 1

            score = relevant_docs / len(context) if context else 0.0

            logger.info(
                f"Context precision computed",
                extra_data={"score": score, "relevant_docs": relevant_docs},
            )

            return min(score, 1.0)
        except Exception as e:
            logger.error(f"Error computing context precision: {str(e)}")
            return 0.0

    def compute_context_recall(
        self, question: str, answer: str, context: List
    ) -> float:
        """
        Compute context recall - does context contain all answer info?

        Args:
            question: User question
            answer: Generated answer
            context: List of context documents

        Returns:
            Recall score (0-1)
        """
        try:
            if not context:
                return 0.0

            full_context = " ".join(
                (
                    getattr(doc, "page_content", str(doc))
                    if hasattr(doc, "page_content")
                    else str(doc)
                )
                for doc in context
            ).lower()

            answer_lower = answer.lower()
            answer_words = answer_lower.split()

            found_words = sum(1 for word in answer_words if word in full_context)
            score = found_words / len(answer_words) if answer_words else 0.0

            logger.info(
                f"Context recall computed",
                extra_data={"score": score, "found_words": found_words},
            )

            return min(score, 1.0)
        except Exception as e:
            logger.error(f"Error computing context recall: {str(e)}")
            return 0.0

    def compute_all_metrics(self, question: str, context: List, answer: str) -> Dict:
        """
        Compute all metrics at once.

        Args:
            question: User question
            context: List of context documents
            answer: Generated answer

        Returns:
            Dict with all metric scores
        """
        try:
            # Convert context string to list if needed
            if isinstance(context, str):
                context = [context]

            metrics = {
                "faithfulness": self.compute_faithfulness(
                    question=question,
                    context=" ".join(
                        (
                            getattr(doc, "page_content", str(doc))
                            if hasattr(doc, "page_content")
                            else str(doc)
                        )
                        for doc in context
                    ),
                    answer=answer,
                ),
                "answer_relevance": self.compute_answer_relevance(
                    question=question, answer=answer
                ),
                "context_precision": self.compute_context_precision(
                    question=question, context=context
                ),
                "context_recall": self.compute_context_recall(
                    question=question, answer=answer, context=context
                ),
            }

            logger.info(
                f"All metrics computed",
                extra_data={"metrics_count": len(metrics)},
            )

            return metrics
        except Exception as e:
            logger.error(f"Error computing all metrics: {str(e)}")
            return {
                "faithfulness": 0.0,
                "answer_relevance": 0.0,
                "context_precision": 0.0,
                "context_recall": 0.0,
                "error": str(e),
            }

    def compute_metrics_batch(self, test_cases: List[Dict]) -> Dict:
        """
        Compute metrics for a batch of test cases.

        Args:
            test_cases: List of dicts with question, context, answer

        Returns:
            Dict with aggregated metrics
        """
        try:
            all_metrics = []

            for i, test_case in enumerate(test_cases):
                metrics = self.compute_all_metrics(
                    question=test_case.get("question", ""),
                    context=test_case.get("context", []),
                    answer=test_case.get("answer", ""),
                )
                all_metrics.append(metrics)

            # Aggregate metrics
            aggregated = {
                "faithfulness": (
                    sum(m.get("faithfulness", 0) for m in all_metrics)
                    / len(all_metrics)
                    if all_metrics
                    else 0.0
                ),
                "answer_relevance": (
                    sum(m.get("answer_relevance", 0) for m in all_metrics)
                    / len(all_metrics)
                    if all_metrics
                    else 0.0
                ),
                "context_precision": (
                    sum(m.get("context_precision", 0) for m in all_metrics)
                    / len(all_metrics)
                    if all_metrics
                    else 0.0
                ),
                "context_recall": (
                    sum(m.get("context_recall", 0) for m in all_metrics)
                    / len(all_metrics)
                    if all_metrics
                    else 0.0
                ),
                "sample_count": len(all_metrics),
            }

            logger.info(
                f"Batch metrics computed",
                extra_data={"sample_count": len(all_metrics)},
            )

            return aggregated
        except Exception as e:
            logger.error(f"Error computing batch metrics: {str(e)}")
            return {
                "faithfulness": 0.0,
                "answer_relevance": 0.0,
                "context_precision": 0.0,
                "context_recall": 0.0,
                "sample_count": 0,
                "error": str(e),
            }


class RegressionTester:
    """
    Performs regression testing to detect performance degradation.

    Stores baseline metrics and compares new runs against them,
    flagging any significant metric regressions.
    """

    def __init__(self, s3_client, bucket_name: str):
        """
        Initialize regression tester.

        Args:
            s3_client: boto3 S3 client
            bucket_name: S3 bucket for storing baselines
        """
        self.s3_client = s3_client
        self.bucket_name = bucket_name
        self.baseline_key = "evaluation/baseline_metrics.json"
        self.baseline_metrics = None

        logger.info(f"RegressionTester initialized", extra_data={"bucket": bucket_name})

    def load_baseline(self) -> bool:
        """
        Load baseline metrics from S3.

        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            response = self.s3_client.get_object(
                Bucket=self.bucket_name, Key=self.baseline_key
            )
            self.baseline_metrics = json.loads(response["Body"].read().decode("utf-8"))

            logger.info(
                f"Baseline metrics loaded from S3",
                extra_data={"keys": list(self.baseline_metrics.keys())},
            )
            return True
        except Exception as e:
            logger.warning(f"Could not load baseline: {str(e)}")
            return False

    def save_baseline(self, metrics: Dict) -> bool:
        """
        Save metrics as new baseline.

        Args:
            metrics: Metrics dict to save as baseline

        Returns:
            True if saved successfully, False otherwise
        """
        try:
            baseline_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "metrics": metrics,
            }

            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=self.baseline_key,
                Body=json.dumps(baseline_data, indent=2),
                ContentType="application/json",
            )

            self.baseline_metrics = metrics
            logger.info(f"Baseline metrics saved to S3")
            return True
        except Exception as e:
            logger.error(f"Error saving baseline: {str(e)}")
            return False

    def detect_regressions(
        self, current_metrics: Dict, threshold_percent: float = 10.0
    ) -> Dict:
        """
        Detect performance regressions compared to baseline.

        Args:
            current_metrics: Current metrics to compare
            threshold_percent: Regression threshold (%)

        Returns:
            Dict with regression analysis
        """
        try:
            if not self.baseline_metrics:
                logger.warning("No baseline metrics available for comparison")
                return {
                    "baseline_available": False,
                    "regressions": [],
                    "improvements": [],
                }

            regressions = []
            improvements = []

            baseline = self.baseline_metrics.get("metrics", {})
            current = current_metrics

            for metric_name, baseline_value in baseline.items():
                if not isinstance(baseline_value, (int, float)):
                    continue

                current_value = current.get(metric_name)
                if current_value is None:
                    continue

                if not isinstance(current_value, (int, float)):
                    continue

                change_percent = abs(
                    (current_value - baseline_value) / baseline_value * 100
                )
                change_direction = (
                    "increased" if current_value > baseline_value else "decreased"
                )

                if change_percent >= threshold_percent:
                    if current_value > baseline_value:
                        # For latency metrics, increase is bad
                        if (
                            "latency" in metric_name.lower()
                            or "time" in metric_name.lower()
                        ):
                            regressions.append(
                                {
                                    "metric": metric_name,
                                    "baseline": baseline_value,
                                    "current": current_value,
                                    "change_percent": change_percent,
                                }
                            )
                        else:
                            improvements.append(
                                {
                                    "metric": metric_name,
                                    "baseline": baseline_value,
                                    "current": current_value,
                                    "change_percent": change_percent,
                                }
                            )
                    else:
                        # For latency metrics, decrease is good
                        if (
                            "latency" in metric_name.lower()
                            or "time" in metric_name.lower()
                        ):
                            improvements.append(
                                {
                                    "metric": metric_name,
                                    "baseline": baseline_value,
                                    "current": current_value,
                                    "change_percent": change_percent,
                                }
                            )
                        else:
                            regressions.append(
                                {
                                    "metric": metric_name,
                                    "baseline": baseline_value,
                                    "current": current_value,
                                    "change_percent": change_percent,
                                }
                            )

            logger.info(
                f"Regression analysis completed",
                extra_data={
                    "regressions_found": len(regressions),
                    "improvements_found": len(improvements),
                },
            )

            return {
                "baseline_available": True,
                "regressions": regressions,
                "improvements": improvements,
                "threshold_percent": threshold_percent,
            }

        except Exception as e:
            logger.error(f"Error detecting regressions: {str(e)}")
            return {"error": str(e), "regressions": [], "improvements": []}
