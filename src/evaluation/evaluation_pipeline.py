"""
Evaluation & Tracing Pipeline Orchestrator (Phase 11)

Main orchestrator for complete evaluation pipeline combining
RAG evaluation, RAGAS metrics, regression testing, and comprehensive reporting.
"""

import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evaluation_components import RAGASMetricsEvaluator, RAGEvaluator, RegressionTester
from evaluation_reporter import EvaluationReporter

from data_ingestion.config import Config
from data_ingestion.logging_config import PipelineLogger

logger = PipelineLogger.create_logger(__name__)

# Try importing LangSmith
try:
    from langchain_core.callbacks import LangChainTracer
    from langsmith import Client as LangSmithClient

    LANGSMITH_AVAILABLE = True
except ImportError:
    logger.warning("LangSmith not available - tracing will be skipped")
    LANGSMITH_AVAILABLE = False


class EvaluationPipeline:
    """
    Complete evaluation and tracing pipeline orchestrator.

    Combines RAG evaluation, RAGAS metrics, regression testing,
    LangSmith tracing, and comprehensive reporting into a unified pipeline.
    """

    def __init__(
        self,
        llm_client,
        s3_client,
        bucket_name: str,
        retrieval_pipeline=None,
        generation_pipeline=None,
        enable_tracing: bool = False,
    ):
        """
        Initialize evaluation pipeline.

        Args:
            llm_client: LLM client for evaluation
            s3_client: boto3 S3 client
            bucket_name: S3 bucket for reports and baselines
            retrieval_pipeline: Retrieval pipeline instance
            generation_pipeline: Generation pipeline instance
            enable_tracing: Whether to enable LangSmith tracing
        """
        self.llm_client = llm_client
        self.s3_client = s3_client
        self.bucket_name = bucket_name
        self.retrieval_pipeline = retrieval_pipeline
        self.generation_pipeline = generation_pipeline
        self.enable_tracing = enable_tracing

        # Initialize components
        self.rag_evaluator = RAGEvaluator(
            llm_client, retrieval_pipeline, generation_pipeline
        )
        self.ragas_evaluator = RAGASMetricsEvaluator()
        self.regression_tester = RegressionTester(s3_client, bucket_name)
        self.reporter = EvaluationReporter(s3_client, bucket_name)

        # Initialize LangSmith if enabled
        self.langsmith_tracer = None
        if enable_tracing and LANGSMITH_AVAILABLE:
            try:
                api_key = os.getenv("LANGSMITH_API_KEY")
                project = os.getenv("LANGSMITH_PROJECT", "rag-evaluation")
                ls_client = LangSmithClient(api_key=api_key)
                self.langsmith_tracer = LangChainTracer(
                    project_name=project, client=ls_client
                )
                logger.info(f"LangSmith tracing enabled", extra={"project": project})
            except Exception as e:
                logger.warning(f"Could not initialize LangSmith: {str(e)}")

        logger.info(
            f"EvaluationPipeline initialized",
            extra={
                "has_retrieval": retrieval_pipeline is not None,
                "has_generation": generation_pipeline is not None,
                "tracing_enabled": self.langsmith_tracer is not None,
            },
        )

    def run_complete_evaluation(
        self,
        evaluation_queries: List[str],
        report_name: str = "rag_evaluation",
        save_report: bool = True,
        check_regressions: bool = False,
        update_baseline: bool = False,
        verbose: bool = True,
    ) -> Dict:
        """
        Execute complete evaluation pipeline.

        Args:
            evaluation_queries: List of queries to evaluate
            report_name: Name for the report
            save_report: Whether to save report to S3
            check_regressions: Whether to check for regressions
            update_baseline: Whether to update baseline metrics
            verbose: Whether to log details

        Returns:
            Comprehensive evaluation results
        """
        start_time = time.time()

        try:
            logger.info(
                f"Phase 11 - Evaluation & Tracing started",
                extra={
                    "num_queries": len(evaluation_queries),
                    "report_name": report_name,
                },
            )

            # Step 1: Batch Evaluation (RAG quality)
            logger.info("Step 1: Running RAG batch evaluation...")
            rag_results = self.rag_evaluator.batch_evaluate(
                evaluation_queries, verbose=verbose
            )

            if verbose:
                logger.info(
                    f"RAG evaluation completed: {len(rag_results)} queries evaluated"
                )

            # Step 2: RAGAS Metrics Evaluation
            logger.info("Step 2: Running RAGAS metrics evaluation...")
            ragas_metrics = self.ragas_evaluator.evaluate(rag_results)

            if verbose:
                self.ragas_evaluator.print_metric_report(ragas_metrics)

            # Step 3: Regression Analysis (if enabled)
            regression_analysis = {}
            if check_regressions:
                logger.info("Step 3: Checking for regressions...")
                baseline_loaded = self.regression_tester.load_baseline()

                if baseline_loaded:
                    # Extract summary metrics for comparison
                    summary_metrics = {
                        "average_latency_ms": self._calculate_avg_latency(rag_results),
                        "success_rate": self._calculate_success_rate(rag_results),
                    }

                    regression_analysis = self.regression_tester.detect_regressions(
                        summary_metrics
                    )

                    if verbose:
                        logger.info(
                            f"Regressions found: {len(regression_analysis.get('regressions', []))}"
                        )
                        logger.info(
                            f"Improvements found: {len(regression_analysis.get('improvements', []))}"
                        )
                else:
                    if verbose:
                        logger.info(
                            "No baseline available - skipping regression analysis"
                        )

            # Step 4: Generate Comprehensive Report
            logger.info("Step 4: Generating comprehensive report...")
            report = self.reporter.generate_report(
                rag_results, ragas_metrics, regression_analysis, report_name
            )

            if verbose:
                self.reporter.print_report(report)

            # Step 5: Update Baseline (if requested)
            if update_baseline:
                logger.info("Step 5: Updating baseline metrics...")
                self.regression_tester.save_baseline(
                    {
                        "average_latency_ms": self._calculate_avg_latency(rag_results),
                        "success_rate": self._calculate_success_rate(rag_results),
                    }
                )
                logger.info("Baseline metrics updated")

            # Step 6: Save Report (if requested)
            if save_report:
                logger.info("Step 6: Saving report to S3...")
                self.reporter.save_report(report, format="json")
                logger.info("Report saved to S3")

            elapsed_time = time.time() - start_time

            # Final Result
            result = {
                "status": "completed",
                "evaluation_timestamp": datetime.utcnow().isoformat(),
                "total_queries": len(evaluation_queries),
                "rag_evaluation_results": rag_results,
                "ragas_metrics": ragas_metrics,
                "regression_analysis": regression_analysis,
                "report": report,
                "execution_time_seconds": elapsed_time,
            }

            logger.info(
                f"Phase 11 - Evaluation & Tracing completed",
                extra={"execution_time": elapsed_time, "success": True},
            )

            return result

        except Exception as e:
            logger.error(f"Error in complete evaluation: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "execution_time_seconds": time.time() - start_time,
            }

    def run_quick_evaluation(
        self, evaluation_queries: List[str], verbose: bool = True
    ) -> Dict:
        """
        Run quick evaluation (RAG only, no regression/baseline).

        Args:
            evaluation_queries: List of queries
            verbose: Whether to log details

        Returns:
            Quick evaluation results
        """
        try:
            start_time = time.time()

            logger.info(
                f"Running quick evaluation",
                extra={"num_queries": len(evaluation_queries)},
            )

            rag_results = self.rag_evaluator.batch_evaluate(
                evaluation_queries, verbose=verbose
            )

            summary = {
                "total_queries": len(rag_results),
                "successful": len([r for r in rag_results if "error" not in r]),
                "failed": len([r for r in rag_results if "error" in r]),
                "average_latency_ms": self._calculate_avg_latency(rag_results),
                "results": rag_results,
                "execution_time_seconds": time.time() - start_time,
            }

            logger.info(
                f"Quick evaluation completed",
                extra={
                    "success_rate": summary["successful"] / len(rag_results) * 100
                    if rag_results
                    else 0
                },
            )

            return summary

        except Exception as e:
            logger.error(f"Error in quick evaluation: {str(e)}", exc_info=True)
            return {"error": str(e), "execution_time_seconds": time.time() - start_time}

    def _calculate_avg_latency(self, results: List[Dict]) -> float:
        """Calculate average latency from results."""
        try:
            valid_results = [r for r in results if "latencies" in r]
            if not valid_results:
                return 0.0

            total_latency = sum(r["latencies"]["total_ms"] for r in valid_results)
            return total_latency / len(valid_results)
        except Exception as e:
            logger.warning(f"Error calculating average latency: {str(e)}")
            return 0.0

    def _calculate_success_rate(self, results: List[Dict]) -> float:
        """Calculate success rate from results."""
        try:
            if not results:
                return 0.0

            successful = len(
                [r for r in results if "error" not in r or r["error"] is None]
            )
            return successful / len(results) * 100
        except Exception as e:
            logger.warning(f"Error calculating success rate: {str(e)}")
            return 0.0


# Statistics collector for evaluation runs
class EvaluationStatistics:
    """Collects statistics across multiple evaluation runs."""

    def __init__(self):
        """Initialize statistics collector."""
        self.total_runs = 0
        self.total_queries_evaluated = 0
        self.total_successful = 0
        self.total_failed = 0
        self.total_execution_time = 0.0
        self.regressions_detected = 0
        self.baseline_updates = 0

        logger.debug("EvaluationStatistics initialized")

    def record_evaluation_run(self, result: Dict):
        """Record evaluation run results."""
        try:
            self.total_runs += 1
            self.total_queries_evaluated += result.get("total_queries", 0)

            rag_results = result.get("rag_evaluation_results", [])
            self.total_successful += len([r for r in rag_results if "error" not in r])
            self.total_failed += len([r for r in rag_results if "error" in r])

            self.total_execution_time += result.get("execution_time_seconds", 0)

            regression = result.get("regression_analysis", {})
            if regression.get("regressions"):
                self.regressions_detected += 1
        except Exception as e:
            logger.warning(f"Error recording evaluation run: {str(e)}")

    def get_summary(self) -> Dict:
        """Get statistics summary."""
        try:
            avg_time = (
                self.total_execution_time / self.total_runs
                if self.total_runs > 0
                else 0
            )
            success_rate = (
                (self.total_successful / self.total_queries_evaluated * 100)
                if self.total_queries_evaluated > 0
                else 0
            )

            summary = {
                "total_evaluation_runs": self.total_runs,
                "total_queries_evaluated": self.total_queries_evaluated,
                "total_successful": self.total_successful,
                "total_failed": self.total_failed,
                "overall_success_rate_percent": success_rate,
                "total_execution_time_seconds": self.total_execution_time,
                "average_execution_time_per_run": avg_time,
                "regressions_detected_count": self.regressions_detected,
            }

            logger.info(f"Evaluation statistics summary", extra=summary)
            return summary
        except Exception as e:
            logger.error(f"Error getting statistics summary: {str(e)}", exc_info=True)
            return {}
