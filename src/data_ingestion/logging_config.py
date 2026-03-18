"""
Logging Configuration Module

Provides CloudWatch-compatible logging for AWS Lambda and cloud deployment.
Supports both JSON and plain text logging formats.
"""

import json
import logging
import sys
from datetime import datetime
from typing import Optional

from pythonjsonlogger import jsonlogger


class CloudWatchFormatter(logging.Formatter):
    """Custom formatter for CloudWatch-compatible logs"""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record for CloudWatch"""
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line_number": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add custom attributes if present
        if hasattr(record, "extra_data"):
            log_data.update(record.extra_data)

        return json.dumps(log_data)


class PipelineLogger:
    """Logger wrapper for data ingestion pipeline"""

    def __init__(
        self,
        name: str = "DataIngestionPipeline",
        level: str = "INFO",
        log_format: str = "json",
        enable_cloudwatch: bool = False,
        cloudwatch_group: Optional[str] = None,
        cloudwatch_stream: Optional[str] = None,
    ):
        """
        Initialize pipeline logger.

        Args:
            name: Logger name
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_format: Log format ('json' or 'plain')
            enable_cloudwatch: Whether to send logs to CloudWatch
            cloudwatch_group: CloudWatch log group name
            cloudwatch_stream: CloudWatch log stream name
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level))

        # Remove existing handlers to avoid duplicate logs
        self.logger.handlers = []

        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level))

        # Set formatter
        if log_format == "json":
            formatter = jsonlogger.JsonFormatter(
                "%(timestamp)s %(level)s %(name)s %(message)s"
            )
        else:
            formatter = logging.Formatter(
                "[%(asctime)s] %(levelname)s [%(name)s:%(funcName)s:%(lineno)d] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )

        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # Add CloudWatch handler if enabled
        if enable_cloudwatch:
            self._add_cloudwatch_handler(
                cloudwatch_group or "/aws/lambda/rag-ingestion",
                cloudwatch_stream or "data-ingestion",
            )

    def _add_cloudwatch_handler(self, log_group: str, log_stream: str) -> None:
        """
        Add CloudWatch handler to logger.

        Args:
            log_group: CloudWatch log group name
            log_stream: CloudWatch log stream name
        """
        try:
            import watchtower

            cloudwatch_handler = watchtower.CloudWatchLogHandler(
                log_group=log_group, stream_name=log_stream
            )
            cloudwatch_handler.setLevel(self.logger.level)
            formatter = CloudWatchFormatter()
            cloudwatch_handler.setFormatter(formatter)
            self.logger.addHandler(cloudwatch_handler)
            self.logger.info(f"CloudWatch handler enabled: {log_group}/{log_stream}")
        except ImportError:
            self.logger.warning(
                "watchtower not installed. CloudWatch logging disabled. "
                "Install with: pip install watchtower"
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize CloudWatch handler: {str(e)}")

    def info(self, message: str, extra_data: Optional[dict] = None) -> None:
        """Log info level message"""
        if extra_data:
            self.logger.info(message, extra={"extra_data": extra_data})
        else:
            self.logger.info(message)

    def warning(self, message: str, extra_data: Optional[dict] = None) -> None:
        """Log warning level message"""
        if extra_data:
            self.logger.warning(message, extra={"extra_data": extra_data})
        else:
            self.logger.warning(message)

    def error(self, message: str, extra_data: Optional[dict] = None) -> None:
        """Log error level message"""
        if extra_data:
            self.logger.error(message, extra={"extra_data": extra_data})
        else:
            self.logger.error(message)

    def debug(self, message: str, extra_data: Optional[dict] = None) -> None:
        """Log debug level message"""
        if extra_data:
            self.logger.debug(message, extra={"extra_data": extra_data})
        else:
            self.logger.debug(message)

    def get_logger(self) -> logging.Logger:
        """Get underlying logger object"""
        return self.logger


def create_logger(
    name: str = "DataIngestionPipeline",
    level: str = "INFO",
    log_format: str = "json",
    enable_cloudwatch: bool = False,
) -> PipelineLogger:
    """
    Factory function to create a configured logger.

    Args:
        name: Logger name
        level: Log level
        log_format: Log format (json or plain)
        enable_cloudwatch: Enable CloudWatch logging

    Returns:
        Configured PipelineLogger instance
    """
    return PipelineLogger(
        name=name,
        level=level,
        log_format=log_format,
        enable_cloudwatch=enable_cloudwatch,
    )
