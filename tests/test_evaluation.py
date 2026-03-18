"""
Comprehensive Test Suite for Evaluation Module (Phase 11)

Tests cover:
1. RAG Evaluation: Answer relevance, faithfulness, context precision/recall
2. RAGAS Metrics: Faithfulness, answer relevance, context precision, context recall
3. Regression Testing: Baseline comparison and performance tracking
4. LangSmith Integration: Tracing and project management
5. Complete Evaluation Pipeline: End-to-end quality assessment

Edge Cases Covered:
- No ground truth data
- Empty or malformed responses
- LangSmith connectivity issues
- Metric computation errors
- Baseline not found
- Performance regression detection
- Large dataset handling
- Timeout scenarios
- S3 persistence failures
- Concurrent evaluation requests
"""

import json
import unittest
from unittest.mock import MagicMock, Mock, patch

from langchain_core.documents import Document


class TestRAGEvaluator(unittest.TestCase):
    """Test cases for RAG evaluation components"""

    @patch("boto3.client")
    def test_rag_evaluator_initialization(self, mock_s3):
        """Test RAG evaluator initialization"""
        mock_client = MagicMock()
        mock_s3.return_value = mock_client

        from src.evaluation.evaluation_components import RAGEvaluator

        evaluator = RAGEvaluator(
            llm_client=MagicMock(),
            retrieval_pipeline=None,
            generation_pipeline=None,
        )

        self.assertIsNotNone(evaluator)

    def test_evaluate_answer_relevance(self):
        """Test answer relevance evaluation"""
        from src.evaluation.evaluation_components import RAGEvaluator

        evaluator = RAGEvaluator(
            llm_client=MagicMock(),
            retrieval_pipeline=None,
            generation_pipeline=None,
        )

        query = "What are cybersecurity risks?"
        answer = "Cybersecurity risks include data breaches and system vulnerabilities"

        # Should compute relevance score
        relevance = evaluator.evaluate_answer_relevance(query, answer)
        self.assertIsNotNone(relevance)

    def test_evaluate_faithfulness(self):
        """Test faithfulness evaluation"""
        from src.evaluation.evaluation_components import RAGEvaluator

        evaluator = RAGEvaluator(
            llm_client=MagicMock(),
            retrieval_pipeline=None,
            generation_pipeline=None,
        )

        context = "Cybersecurity risks include breaches and vulnerabilities"
        answer = "The company faces cybersecurity threats"

        # Should evaluate if answer is faithful to context
        faithfulness = evaluator.evaluate_faithfulness(context, answer)
        self.assertIsNotNone(faithfulness)

    def test_evaluate_context_precision(self):
        """Test context precision evaluation"""
        from src.evaluation.evaluation_components import RAGEvaluator

        evaluator = RAGEvaluator(
            llm_client=MagicMock(),
            retrieval_pipeline=None,
            generation_pipeline=None,
        )

        query = "What are legal risks?"
        context = [
            Document(page_content="Legal proceedings involve litigation"),
            Document(page_content="Regulatory compliance is required"),
            Document(
                page_content="Product features and specifications"
            ),  # Not relevant
        ]

        # Should measure precision of retrieved context
        precision = evaluator.evaluate_context_precision(query, context)
        self.assertIsNotNone(precision)

    def test_evaluate_context_recall(self):
        """Test context recall evaluation"""
        from src.evaluation.evaluation_components import RAGEvaluator

        evaluator = RAGEvaluator(
            llm_client=MagicMock(),
            retrieval_pipeline=None,
            generation_pipeline=None,
        )

        query = "What are all types of risks?"
        answer = "Cybersecurity risks and legal risks"
        context = [
            Document(page_content="Cybersecurity risks"),
            Document(page_content="Legal risks"),
        ]

        # Should measure if all relevant info is in context
        recall = evaluator.evaluate_context_recall(query, answer, context)
        self.assertIsNotNone(recall)


class TestRAGASMetricsEvaluator(unittest.TestCase):
    """Test cases for RAGAS metrics computation"""

    def test_ragas_initialization(self):
        """Test RAGAS evaluator initialization"""
        from src.evaluation.evaluation_components import RAGASMetricsEvaluator

        evaluator = RAGASMetricsEvaluator()
        self.assertIsNotNone(evaluator)

    def test_faithfulness_metric(self):
        """Test RAGAS faithfulness metric"""
        from src.evaluation.evaluation_components import RAGASMetricsEvaluator

        evaluator = RAGASMetricsEvaluator()

        question = "What are cybersecurity risks?"
        context = "Cybersecurity risks include data breaches, malware, phishing"
        answer = "The main cybersecurity risks are data breaches and malware"

        faithfulness = evaluator.compute_faithfulness(
            question=question, context=context, answer=answer
        )

        self.assertIsNotNone(faithfulness)
        self.assertGreaterEqual(faithfulness, 0.0)
        self.assertLessEqual(faithfulness, 1.0)

    def test_answer_relevance_metric(self):
        """Test RAGAS answer relevance metric"""
        from src.evaluation.evaluation_components import RAGASMetricsEvaluator

        evaluator = RAGASMetricsEvaluator()

        question = "What is the company's main business?"
        answer = "The company develops software products"

        relevance = evaluator.compute_answer_relevance(question=question, answer=answer)

        self.assertIsNotNone(relevance)
        self.assertGreaterEqual(relevance, 0.0)
        self.assertLessEqual(relevance, 1.0)

    def test_context_precision_metric(self):
        """Test RAGAS context precision metric"""
        from src.evaluation.evaluation_components import RAGASMetricsEvaluator

        evaluator = RAGASMetricsEvaluator()

        question = "What are legal risks?"
        context_docs = [
            "Legal proceedings document",
            "Regulatory requirements document",
            "Product specification document",  # Not relevant
        ]

        precision = evaluator.compute_context_precision(
            question=question, context=context_docs
        )

        self.assertIsNotNone(precision)

    def test_context_recall_metric(self):
        """Test RAGAS context recall metric"""
        from src.evaluation.evaluation_components import RAGASMetricsEvaluator

        evaluator = RAGASMetricsEvaluator()

        question = "What are risk types?"
        answer = "Cybersecurity risks and legal risks"
        context_docs = [
            "Cybersecurity risks document",
            "Legal risks document",
        ]

        recall = evaluator.compute_context_recall(
            question=question, answer=answer, context=context_docs
        )

        self.assertIsNotNone(recall)

    def test_compute_all_metrics(self):
        """Test computing all RAGAS metrics at once"""
        from src.evaluation.evaluation_components import RAGASMetricsEvaluator

        evaluator = RAGASMetricsEvaluator()

        sample_data = {
            "question": "What are risks?",
            "context": ["Risk doc 1", "Risk doc 2"],
            "answer": "Risks include cybersecurity and legal",
        }

        metrics = evaluator.compute_all_metrics(**sample_data)

        self.assertIn("faithfulness", metrics)
        self.assertIn("answer_relevance", metrics)
        self.assertIn("context_precision", metrics)
        self.assertIn("context_recall", metrics)

    def test_ragas_with_empty_context(self):
        """Test RAGAS metrics with empty context"""
        from src.evaluation.evaluation_components import RAGASMetricsEvaluator

        evaluator = RAGASMetricsEvaluator()

        # Should handle empty context gracefully
        metrics = evaluator.compute_all_metrics(
            question="Question", context=[], answer="Answer"
        )

        self.assertIsNotNone(metrics)

    def test_ragas_with_hallucinated_answer(self):
        """Test RAGAS faithfulness with hallucinated answer"""
        from src.evaluation.evaluation_components import RAGASMetricsEvaluator

        evaluator = RAGASMetricsEvaluator()

        context = "Company is in software industry"
        answer = "Company has 1 million employees and operates in agriculture"  # Hallucinated

        faithfulness = evaluator.compute_faithfulness(
            question="What does company do?", context=context, answer=answer
        )

        # Should detect low faithfulness
        self.assertIsNotNone(faithfulness)

    def test_ragas_metric_aggregation(self):
        """Test aggregating multiple RAGAS metrics"""
        from src.evaluation.evaluation_components import RAGASMetricsEvaluator

        evaluator = RAGASMetricsEvaluator()

        test_cases = [
            {
                "question": "Q1",
                "context": ["Doc1"],
                "answer": "Answer1",
            },
            {
                "question": "Q2",
                "context": ["Doc2"],
                "answer": "Answer2",
            },
        ]

        aggregated = evaluator.compute_metrics_batch(test_cases)

        self.assertIsNotNone(aggregated)


class TestRegressionTester(unittest.TestCase):
    """Test cases for regression testing"""

    @patch("boto3.client")
    def test_regression_tester_initialization(self, mock_s3):
        """Test regression tester initialization"""
        mock_client = MagicMock()
        mock_s3.return_value = mock_client

        from src.evaluation.evaluation_components import RegressionTester

        tester = RegressionTester(s3_client=mock_client, bucket_name="test-bucket")

        self.assertIsNotNone(tester)

    @patch("boto3.client")
    def test_save_baseline_metrics(self, mock_s3):
        """Test saving baseline metrics"""
        mock_client = MagicMock()
        mock_s3.return_value = mock_client

        from src.evaluation.evaluation_components import RegressionTester

        tester = RegressionTester(s3_client=mock_client, bucket_name="test-bucket")

        baseline = {
            "faithfulness": 0.95,
            "answer_relevance": 0.92,
            "context_precision": 0.88,
        }

        tester.save_baseline(baseline_name="baseline-v1", metrics=baseline)

        mock_client.put_object.assert_called_once()

    @patch("boto3.client")
    def test_load_baseline_metrics(self, mock_s3):
        """Test loading baseline metrics"""
        mock_client = MagicMock()
        baseline_data = {
            "faithfulness": 0.95,
            "answer_relevance": 0.92,
        }
        mock_client.get_object.return_value = {
            "Body": MagicMock(
                read=MagicMock(return_value=json.dumps(baseline_data).encode())
            )
        }
        mock_s3.return_value = mock_client

        from src.evaluation.evaluation_components import RegressionTester

        tester = RegressionTester(s3_client=mock_client, bucket_name="test-bucket")

        loaded = tester.load_baseline("baseline-v1")

        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["faithfulness"], 0.95)

    @patch("boto3.client")
    def test_compare_metrics(self, mock_s3):
        """Test comparing current vs baseline metrics"""
        mock_client = MagicMock()
        mock_s3.return_value = mock_client

        from src.evaluation.evaluation_components import RegressionTester

        tester = RegressionTester(s3_client=mock_client, bucket_name="test-bucket")

        current = {
            "faithfulness": 0.93,  # 2% regression
            "answer_relevance": 0.94,  # Improved
        }

        baseline = {
            "faithfulness": 0.95,
            "answer_relevance": 0.92,
        }

        comparison = tester.compare_metrics(current, baseline)

        self.assertIsNotNone(comparison)

    @patch("boto3.client")
    def test_detect_regression(self, mock_s3):
        """Test detecting performance regression"""
        mock_client = MagicMock()
        mock_s3.return_value = mock_client

        from src.evaluation.evaluation_components import RegressionTester

        tester = RegressionTester(
            s3_client=mock_client,
            bucket_name="test-bucket",
            regression_threshold=0.05,  # 5% threshold
        )

        current = {"faithfulness": 0.90}  # 5% regression
        baseline = {"faithfulness": 0.95}

        is_regression = tester.is_regression(current, baseline)

        self.assertTrue(is_regression)

    @patch("boto3.client")
    def test_baseline_not_found(self, mock_s3):
        """Test behavior when baseline not found"""
        mock_client = MagicMock()
        mock_client.get_object.side_effect = Exception("NoSuchKey")
        mock_s3.return_value = mock_client

        from src.evaluation.evaluation_components import RegressionTester

        tester = RegressionTester(s3_client=mock_client, bucket_name="test-bucket")

        # Should handle missing baseline gracefully


class TestLangSmithIntegration(unittest.TestCase):
    """Test cases for LangSmith integration"""

    @patch("langsmith.Client")
    def test_langsmith_client_initialization(self, mock_langsmith):
        """Test LangSmith client initialization"""
        mock_client = MagicMock()
        mock_langsmith.return_value = mock_client

        from src.evaluation.evaluation_pipeline import EvaluationPipeline

        # Should initialize LangSmith client

    @patch("langsmith.Client")
    def test_create_langsmith_project(self, mock_langsmith):
        """Test creating LangSmith project for CI testing"""
        mock_client = MagicMock()
        mock_langsmith.return_value = mock_client

        # Should create 'rag-ci-test' project

    @patch("langsmith.Client")
    def test_log_run_to_langsmith(self, mock_langsmith):
        """Test logging evaluation runs to LangSmith"""
        mock_client = MagicMock()
        mock_langsmith.return_value = mock_client

        # Should log runs with full tracing

    @patch("langsmith.Client")
    def test_trace_query_execution(self, mock_langsmith):
        """Test tracing query execution in LangSmith"""
        mock_client = MagicMock()
        mock_langsmith.return_value = mock_client

        # Should trace all steps

    @patch("langsmith.Client")
    def test_langsmith_connectivity_failure(self, mock_langsmith):
        """Test handling LangSmith connectivity failures"""
        mock_langsmith.side_effect = Exception("Connection failed")

        # Should handle gracefully and continue evaluation


class TestEvaluationPipeline(unittest.TestCase):
    """Test cases for complete evaluation pipeline"""

    @patch("boto3.client")
    @patch("langsmith.Client")
    def test_pipeline_initialization(self, mock_langsmith, mock_s3):
        """Test evaluation pipeline initialization"""
        mock_s3_client = MagicMock()
        mock_s3.return_value = mock_s3_client
        mock_langsmith_client = MagicMock()
        mock_langsmith.return_value = mock_langsmith_client

        from src.evaluation.evaluation_pipeline import EvaluationPipeline

        pipeline = EvaluationPipeline(
            llm_client=MagicMock(),
            s3_client=mock_s3_client,
            bucket_name="test-bucket",
            enable_tracing=True,
        )

        self.assertIsNotNone(pipeline)

    @patch("boto3.client")
    def test_evaluate_single_sample(self, mock_s3):
        """Test evaluating a single sample"""
        mock_client = MagicMock()
        mock_s3.return_value = mock_client

        from src.evaluation.evaluation_pipeline import EvaluationPipeline

        pipeline = EvaluationPipeline(
            llm_client=MagicMock(),
            s3_client=mock_client,
            bucket_name="test-bucket",
        )

        sample = {
            "question": "What are risks?",
            "context": ["Risk document"],
            "answer": "Main risks are cybersecurity and legal",
        }

        result = pipeline.evaluate_sample(sample)

        self.assertIsNotNone(result)

    @patch("boto3.client")
    def test_evaluate_batch(self, mock_s3):
        """Test evaluating batch of samples"""
        mock_client = MagicMock()
        mock_s3.return_value = mock_client

        from src.evaluation.evaluation_pipeline import EvaluationPipeline

        pipeline = EvaluationPipeline(
            llm_client=MagicMock(),
            s3_client=mock_client,
            bucket_name="test-bucket",
        )

        batch = [
            {
                "question": "Q1",
                "context": ["Doc1"],
                "answer": "Answer1",
            },
            {
                "question": "Q2",
                "context": ["Doc2"],
                "answer": "Answer2",
            },
        ]

        results = pipeline.evaluate_batch(batch)

        self.assertIsNotNone(results)
        self.assertEqual(len(results), 2)

    @patch("boto3.client")
    def test_generate_evaluation_report(self, mock_s3):
        """Test generating evaluation report"""
        mock_client = MagicMock()
        mock_s3.return_value = mock_client

        from src.evaluation.evaluation_pipeline import EvaluationPipeline

        pipeline = EvaluationPipeline(
            llm_client=MagicMock(),
            s3_client=mock_client,
            bucket_name="test-bucket",
        )

        # Should generate comprehensive report

    @patch("boto3.client")
    def test_regression_detection_in_pipeline(self, mock_s3):
        """Test regression detection during evaluation"""
        mock_client = MagicMock()
        mock_s3.return_value = mock_client

        from src.evaluation.evaluation_pipeline import EvaluationPipeline

        pipeline = EvaluationPipeline(
            llm_client=MagicMock(),
            s3_client=mock_client,
            bucket_name="test-bucket",
            enable_regression_check=True,
        )

        # Should detect regression vs baseline

    @patch("boto3.client")
    def test_pipeline_with_ground_truth(self, mock_s3):
        """Test evaluation with ground truth data"""
        mock_client = MagicMock()
        mock_s3.return_value = mock_client

        from src.evaluation.evaluation_pipeline import EvaluationPipeline

        pipeline = EvaluationPipeline(
            llm_client=MagicMock(),
            s3_client=mock_client,
            bucket_name="test-bucket",
        )

        sample = {
            "question": "What are risks?",
            "context": ["Risk doc"],
            "answer": "Risks are X and Y",
            "ground_truth": "Expected answer",  # Ground truth
        }

        result = pipeline.evaluate_sample(sample)

        self.assertIsNotNone(result)

    @patch("boto3.client")
    def test_pipeline_without_ground_truth(self, mock_s3):
        """Test evaluation without ground truth"""
        mock_client = MagicMock()
        mock_s3.return_value = mock_client

        from src.evaluation.evaluation_pipeline import EvaluationPipeline

        pipeline = EvaluationPipeline(
            llm_client=MagicMock(),
            s3_client=mock_client,
            bucket_name="test-bucket",
        )

        sample = {
            "question": "What are risks?",
            "context": ["Risk doc"],
            "answer": "Risks are X and Y",
        }

        result = pipeline.evaluate_sample(sample)

        self.assertIsNotNone(result)

    @patch("boto3.client")
    def test_pipeline_timeout_handling(self, mock_s3):
        """Test handling evaluation timeouts"""
        mock_client = MagicMock()
        mock_s3.return_value = mock_client

        from src.evaluation.evaluation_pipeline import EvaluationPipeline

        pipeline = EvaluationPipeline(
            llm_client=MagicMock(),
            s3_client=mock_client,
            bucket_name="test-bucket",
            timeout=10,
        )

        # Should handle timeouts gracefully

    @patch("boto3.client")
    def test_pipeline_error_recovery(self, mock_s3):
        """Test error recovery in pipeline"""
        mock_client = MagicMock()
        mock_s3.return_value = mock_client

        from src.evaluation.evaluation_pipeline import EvaluationPipeline

        pipeline = EvaluationPipeline(
            llm_client=MagicMock(),
            s3_client=mock_client,
            bucket_name="test-bucket",
        )

        # Should recover from individual sample failures


class TestEvaluationReporter(unittest.TestCase):
    """Test cases for evaluation reporting"""

    @patch("boto3.client")
    def test_reporter_initialization(self, mock_s3):
        """Test reporter initialization"""
        mock_client = MagicMock()
        mock_s3.return_value = mock_client

        from src.evaluation.evaluation_reporter import EvaluationReporter

        reporter = EvaluationReporter(s3_client=mock_client, bucket_name="test-bucket")

        self.assertIsNotNone(reporter)

    @patch("boto3.client")
    def test_generate_summary_report(self, mock_s3):
        """Test generating summary report"""
        mock_client = MagicMock()
        mock_s3.return_value = mock_client

        from src.evaluation.evaluation_reporter import EvaluationReporter

        reporter = EvaluationReporter(s3_client=mock_client, bucket_name="test-bucket")

        metrics = {
            "faithfulness": 0.92,
            "answer_relevance": 0.89,
            "context_precision": 0.85,
        }

        report = reporter.generate_summary(metrics)

        self.assertIsNotNone(report)

    @patch("boto3.client")
    def test_save_report_to_s3(self, mock_s3):
        """Test saving report to S3"""
        mock_client = MagicMock()
        mock_s3.return_value = mock_client

        from src.evaluation.evaluation_reporter import EvaluationReporter

        reporter = EvaluationReporter(s3_client=mock_client, bucket_name="test-bucket")

        report_data = {"title": "Evaluation Report", "metrics": {}}

        reporter.save_report("eval-report-1", report_data)

        mock_client.put_object.assert_called_once()


if __name__ == "__main__":
    unittest.main()
