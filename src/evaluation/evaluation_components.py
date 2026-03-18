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
from data_ingestion.logging_config import PipelineLogger

logger = PipelineLogger.create_logger(__name__)

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
            extra={
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
                            extra={
                                "latency_ms": result["latencies"]["retrieval_ms"],
                                "docs_retrieved": len(result["retrieved_docs"]),
                            },
                        )
                except Exception as e:
                    logger.error(f"Retrieval failed: {str(e)}", exc_info=True)
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
                            extra={
                                "latency_ms": result["latencies"]["generation_ms"],
                                "response_length": len(result["response"]),
                            },
                        )
                except Exception as e:
                    logger.error(f"Generation failed: {str(e)}", exc_info=True)
                    result["error"] = f"Generation error: {str(e)}"
                    result["latencies"]["total_ms"] = (time.time() - start_time) * 1000
                    return result

            result["latencies"]["total_ms"] = (time.time() - start_time) * 1000

            logger.debug(
                f"Query evaluation completed",
                extra={"total_latency_ms": result["latencies"]["total_ms"]},
            )

            return result

        except Exception as e:
            logger.error(f"Error evaluating query: {str(e)}", exc_info=True)
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
        logger.info(f"Starting batch evaluation", extra={"query_count": len(queries)})

        results = []

        for i, query in enumerate(queries, 1):
            try:
                if verbose:
                    logger.info(f"Evaluating query {i}/{len(queries)}: {query[:50]}...")

                result = self.evaluate_single_query(query, verbose=verbose)
                results.append(result)

            except Exception as e:
                logger.error(
                    f"Error in batch evaluation for query {i}: {str(e)}", exc_info=True
                )
                results.append(
                    {"query": query, "error": str(e), "latencies": {"total_ms": 0}}
                )

        logger.info(
            f"Batch evaluation completed",
            extra={
                "total_queries": len(queries),
                "successful": len([r for r in results if "error" not in r]),
                "failed": len([r for r in results if "error" in r]),
            },
        )

        return results


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
            extra={"metrics_available": len(self.metrics)},
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
                extra={"samples": len(dataset), "columns": dataset.column_names},
            )

            return dataset
        except Exception as e:
            logger.error(f"Error preparing dataset: {str(e)}", exc_info=True)
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
                extra={"metrics_computed": len(overall_scores)},
            )

            return result

        except Exception as e:
            logger.error(f"Error during RAGAS evaluation: {str(e)}", exc_info=True)
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
            logger.error(f"Error printing metrics report: {str(e)}", exc_info=True)


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

        logger.info(f"RegressionTester initialized", extra={"bucket": bucket_name})

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
                extra={"keys": list(self.baseline_metrics.keys())},
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
            logger.error(f"Error saving baseline: {str(e)}", exc_info=True)
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
                extra={
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
            logger.error(f"Error detecting regressions: {str(e)}", exc_info=True)
            return {"error": str(e), "regressions": [], "improvements": []}
