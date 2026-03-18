"""
Evaluation Reporter for Evaluation & Tracing Pipeline

Generates comprehensive evaluation reports with latency analysis,
recommendations, and S3 persistence.
"""

import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_ingestion.logging_config import create_logger

logger = create_logger(__name__)


class EvaluationReporter:
    """
    Generate comprehensive evaluation reports with visualizations.

    Creates detailed reports including summary statistics, metrics,
    latency analysis, regression analysis, and recommendations.
    """

    def __init__(self, s3_client, bucket_name: str):
        """
        Initialize evaluation reporter.

        Args:
            s3_client: boto3 S3 client
            bucket_name: S3 bucket for storing reports
        """
        self.s3_client = s3_client
        self.bucket_name = bucket_name

        logger.info(
            f"EvaluationReporter initialized", extra_data={"bucket": bucket_name}
        )

    def generate_report(
        self,
        rag_evaluator_results: List[Dict],
        ragas_metrics: Dict,
        regression_analysis: Dict = None,
        report_name: str = "rag_evaluation",
    ) -> Dict:
        """
        Generate comprehensive evaluation report.

        Args:
            rag_evaluator_results: List of evaluation results from RAGEvaluator
            ragas_metrics: RAGAS metrics dict from RAGASMetricsEvaluator
            regression_analysis: Optional regression analysis results
            report_name: Name for the report

        Returns:
            Comprehensive report dict
        """
        try:
            logger.info(
                f"Generating evaluation report: {report_name}",
                extra_data={"results_count": len(rag_evaluator_results)},
            )

            report = {
                "timestamp": datetime.utcnow().isoformat(),
                "report_name": report_name,
                "summary": self._generate_summary(rag_evaluator_results, ragas_metrics),
                "metrics": ragas_metrics,
                "latency_analysis": self._analyze_latencies(rag_evaluator_results),
                "regression_analysis": regression_analysis or {},
                "recommendations": self._generate_recommendations(ragas_metrics),
                "result_count": len(rag_evaluator_results),
            }

            logger.info(f"Report generated successfully")
            return report

        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            return {
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
                "report_name": report_name,
            }

    def generate_summary(self, metrics: Dict) -> Dict:
        """
        Generate summary report from metrics.

        Args:
            metrics: Metrics dict with evaluation scores

        Returns:
            Summary report dict
        """
        try:
            logger.info(
                f"Generating summary report",
                extra_data={"metric_keys": list(metrics.keys())},
            )

            summary = {
                "timestamp": datetime.utcnow().isoformat(),
                "metrics": metrics,
                "average_score": sum(
                    v for v in metrics.values() if isinstance(v, (int, float))
                )
                / len([v for v in metrics.values() if isinstance(v, (int, float))])
                if any(isinstance(v, (int, float)) for v in metrics.values())
                else 0.0,
            }

            logger.info(f"Summary report generated successfully")
            return summary
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return {"error": str(e)}

    def _generate_summary(self, results: List[Dict], metrics: Dict) -> Dict:
        """
        Generate summary statistics.

        Args:
            results: Evaluation results
            metrics: RAGAS metrics

        Returns:
            Summary dict
        """
        try:
            if not results:
                return {
                    "total_queries_evaluated": 0,
                    "average_latency_ms": 0,
                    "error": "No results provided",
                }

            # Filter out results with errors
            valid_results = [
                r for r in results if "error" not in r or r["error"] is None
            ]

            if not valid_results:
                return {
                    "total_queries_evaluated": len(results),
                    "failed_queries": len(results),
                    "error": "All queries failed",
                }

            retrieval_times = [
                r["latencies"]["retrieval_ms"]
                for r in valid_results
                if r["latencies"]["retrieval_ms"] > 0
            ]
            generation_times = [
                r["latencies"]["generation_ms"]
                for r in valid_results
                if r["latencies"]["generation_ms"] > 0
            ]
            total_times = [r["latencies"]["total_ms"] for r in valid_results]

            avg_retrieval = (
                sum(retrieval_times) / len(retrieval_times) if retrieval_times else 0
            )
            avg_generation = (
                sum(generation_times) / len(generation_times) if generation_times else 0
            )

            summary = {
                "total_queries_evaluated": len(results),
                "successful_queries": len(valid_results),
                "failed_queries": len(results) - len(valid_results),
                "success_rate_percent": (len(valid_results) / len(results) * 100)
                if results
                else 0,
                "average_latency_ms": sum(total_times) / len(total_times)
                if total_times
                else 0,
                "average_retrieval_latency_ms": avg_retrieval,
                "average_generation_latency_ms": avg_generation,
                "evaluation_date": datetime.utcnow().isoformat(),
            }

            logger.debug(
                f"Summary generated",
                extra_data={"success_rate": summary["success_rate_percent"]},
            )

            return summary
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return {"error": str(e)}

    def _analyze_latencies(self, results: List[Dict]) -> Dict:
        """
        Analyze retrieval and generation latencies.

        Args:
            results: Evaluation results

        Returns:
            Latency analysis dict
        """
        try:
            valid_results = [
                r for r in results if "error" not in r or r["error"] is None
            ]

            if not valid_results:
                return {
                    "error": "No valid results for latency analysis",
                    "retrieval": {},
                    "generation": {},
                    "total": {},
                }

            retrieval_times = [
                r["latencies"]["retrieval_ms"]
                for r in valid_results
                if r["latencies"]["retrieval_ms"] > 0
            ]
            generation_times = [
                r["latencies"]["generation_ms"]
                for r in valid_results
                if r["latencies"]["generation_ms"] > 0
            ]
            total_times = [r["latencies"]["total_ms"] for r in valid_results]

            def get_stats(times):
                """Calculate statistics for a list of times."""
                if not times:
                    return {}
                sorted_times = sorted(times)
                return {
                    "mean_ms": sum(times) / len(times),
                    "min_ms": min(times),
                    "max_ms": max(times),
                    "median_ms": sorted_times[len(sorted_times) // 2],
                    "p95_ms": sorted_times[int(len(sorted_times) * 0.95)]
                    if len(sorted_times) > 1
                    else sorted_times[0],
                    "p99_ms": sorted_times[int(len(sorted_times) * 0.99)]
                    if len(sorted_times) > 1
                    else sorted_times[0],
                    "count": len(times),
                }

            latency_analysis = {
                "retrieval": get_stats(retrieval_times),
                "generation": get_stats(generation_times),
                "total": get_stats(total_times),
            }

            logger.debug(f"Latency analysis completed")
            return latency_analysis

        except Exception as e:
            logger.error(f"Error analyzing latencies: {str(e)}")
            return {"error": str(e)}

    def _generate_recommendations(self, metrics: Dict) -> List[str]:
        """
        Generate actionable recommendations based on metrics.

        Args:
            metrics: RAGAS metrics dict

        Returns:
            List of recommendation strings
        """
        try:
            recommendations = []
            scores = metrics.get("overall_scores", {})

            # Check for low metric scores
            if scores.get("faithfulness", 1) and scores.get("faithfulness") < 0.7:
                recommendations.append(
                    "Low faithfulness score - Review prompt engineering and context selection to improve answer groundedness"
                )

            if (
                scores.get("answer_relevancy", 1)
                and scores.get("answer_relevancy") < 0.7
            ):
                recommendations.append(
                    "Low answer relevancy score - Improve query routing and multi-query generation strategies"
                )

            if (
                scores.get("context_precision", 1)
                and scores.get("context_precision") < 0.7
            ):
                recommendations.append(
                    "Low context precision score - Enhance retrieval reranking strategy or embedding model"
                )

            if scores.get("context_recall", 1) and scores.get("context_recall") < 0.7:
                recommendations.append(
                    "Low context recall score - Increase top_k parameter or improve embedding model quality"
                )

            # If no recommendations, provide positive feedback
            if not recommendations:
                recommendations.append(
                    "✓ All metrics are healthy. Continue monitoring for performance regressions."
                )

            logger.debug(f"Generated {len(recommendations)} recommendations")
            return recommendations

        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            return ["⚠ Error generating recommendations"]

    def save_report(
        self, report_name_or_dict, report_data: Dict = None, format: str = "json"
    ) -> bool:
        """
        Save report to S3.

        Supports two calling signatures:
        1. save_report(report_name: str, report_data: Dict) - new interface
        2. save_report(report: Dict, format: str) - original interface

        Args:
            report_name_or_dict: Report name (string) or report dict
            report_data: Report dict (when first arg is a string) or format string (when first arg is dict)
            format: Format (json or csv) - only used with dict first arg

        Returns:
            True if saved successfully, False otherwise
        """
        try:
            # Handle two different calling signatures
            if isinstance(report_name_or_dict, str) and isinstance(report_data, dict):
                # New interface: save_report(report_name, report_data)
                report = report_data.copy()
                report["report_name"] = report_name_or_dict
                actual_format = format
            elif isinstance(report_name_or_dict, dict):
                # Original interface: save_report(report_dict, format)
                report = report_name_or_dict
                actual_format = report_data if isinstance(report_data, str) else format
            else:
                logger.error("Invalid arguments to save_report")
                return False

            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            report_name = report.get("report_name", "evaluation")
            filename = f"rag-reports/{report_name}_{timestamp}.{actual_format}"

            if actual_format == "json":
                body = json.dumps(report, indent=2, default=str)
                content_type = "application/json"
            else:
                # For CSV, flatten the dict
                import csv
                import io

                output = io.StringIO()
                # Simple flattening for CSV
                writer = csv.DictWriter(output, fieldnames=report.keys())
                writer.writeheader()
                flat_report = {
                    k: str(v)[:100] for k, v in report.items()
                }  # Truncate for CSV
                writer.writerow(flat_report)
                body = output.getvalue()
                content_type = "text/csv"

            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=filename,
                Body=body,
                ContentType=content_type,
            )

            logger.info(
                f"Report saved to S3",
                extra_data={"s3_key": filename, "format": actual_format},
            )
            return True
        except Exception as e:
            logger.error(f"Error saving report to S3: {str(e)}")
            return False

    def print_report(self, report: Dict):
        """
        Print formatted report to logger.

        Args:
            report: Report dict to print
        """
        try:
            logger.info("=" * 80)
            logger.info("RAG EVALUATION REPORT")
            logger.info("=" * 80)

            logger.info(f"Timestamp: {report.get('timestamp', 'N/A')}")

            # Summary Section
            logger.info("--- SUMMARY ---")
            summary = report.get("summary", {})
            if not summary.get("error"):
                logger.info(
                    f"Queries Evaluated: {summary.get('total_queries_evaluated', 0)}"
                )
                logger.info(
                    f"Success Rate: {summary.get('success_rate_percent', 0):.1f}%"
                )
                logger.info(
                    f"Average Total Latency: {summary.get('average_latency_ms', 0):.0f}ms"
                )
                logger.info(
                    f"  - Retrieval: {summary.get('average_retrieval_latency_ms', 0):.0f}ms"
                )
                logger.info(
                    f"  - Generation: {summary.get('average_generation_latency_ms', 0):.0f}ms"
                )
            else:
                logger.warning(f"Summary error: {summary.get('error')}")

            # Metrics Section
            logger.info("--- RAGAS METRICS ---")
            metrics = report.get("metrics", {})
            overall_scores = metrics.get("overall_scores", {})
            for metric_name, score in overall_scores.items():
                if score is not None:
                    logger.info(f"{metric_name}: {score:.4f}")
                else:
                    logger.info(f"{metric_name}: N/A")

            # Latency Analysis
            logger.info("--- LATENCY ANALYSIS ---")
            latency = report.get("latency_analysis", {})
            for latency_type in ["retrieval", "generation", "total"]:
                if latency_type in latency and latency[latency_type]:
                    stats = latency[latency_type]
                    logger.info(f"{latency_type.capitalize()}:")
                    logger.info(f"  Mean: {stats.get('mean_ms', 0):.0f}ms")
                    logger.info(
                        f"  Min: {stats.get('min_ms', 0):.0f}ms, Max: {stats.get('max_ms', 0):.0f}ms"
                    )
                    logger.info(
                        f"  P95: {stats.get('p95_ms', 0):.0f}ms, P99: {stats.get('p99_ms', 0):.0f}ms"
                    )

            # Regression Analysis
            regression = report.get("regression_analysis", {})
            if regression and regression.get("regressions"):
                logger.warning("--- REGRESSIONS DETECTED ---")
                for reg in regression["regressions"]:
                    logger.warning(
                        f"{reg.get('metric')}: {reg.get('change_percent'):.1f}% change"
                    )

            if regression and regression.get("improvements"):
                logger.info("--- IMPROVEMENTS DETECTED ---")
                for imp in regression["improvements"]:
                    logger.info(
                        f"{imp.get('metric')}: {imp.get('change_percent'):.1f}% improvement"
                    )

            # Recommendations
            logger.info("--- RECOMMENDATIONS ---")
            recommendations = report.get("recommendations", [])
            for i, rec in enumerate(recommendations, 1):
                logger.info(f"{i}. {rec}")

            logger.info("=" * 80)

        except Exception as e:
            logger.error(f"Error printing report: {str(e)}")
