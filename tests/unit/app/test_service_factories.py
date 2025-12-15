"""Tests for service factory functions."""

import unittest
from unittest.mock import MagicMock, patch

from src.app.service_factories import (
    get_analytics_service,
    get_inference_service,
    get_training_service,
    get_workflow_service,
)
from src.services.analytics_service import AnalyticsService
from src.services.inference_service import InferenceService
from src.services.training_service import TrainingService


class TestServiceFactories(unittest.TestCase):
    """Tests for service factory functions."""

    @patch("src.app.service_factories.st")
    def test_get_training_service(self, mock_st):
        """Verify get_training_service returns TrainingService instance."""
        service = get_training_service()
        self.assertIsInstance(service, TrainingService)

        # Verify it's cached (same instance on second call)
        service2 = get_training_service()
        self.assertIs(service, service2)

    @patch("src.app.service_factories.st")
    def test_get_inference_service(self, mock_st):
        """Verify get_inference_service returns InferenceService instance."""
        service = get_inference_service()
        self.assertIsInstance(service, InferenceService)

        # Verify it's cached (same instance on second call)
        service2 = get_inference_service()
        self.assertIs(service, service2)

    @patch("src.app.service_factories.st")
    def test_get_analytics_service(self, mock_st):
        """Verify get_analytics_service returns AnalyticsService instance."""
        service = get_analytics_service()
        self.assertIsInstance(service, AnalyticsService)

        # Verify it's cached (same instance on second call)
        service2 = get_analytics_service()
        self.assertIs(service, service2)

    def test_get_workflow_service_default_provider(self):
        """Verify get_workflow_service returns WorkflowService with default provider."""
        with patch("src.app.service_factories.WorkflowService") as mock_workflow_class:
            mock_service = MagicMock()
            mock_workflow_class.return_value = mock_service

            service = get_workflow_service()

            mock_workflow_class.assert_called_once_with(provider="openai", model=None)
            self.assertEqual(service, mock_service)

    def test_get_workflow_service_custom_provider(self):
        """Verify get_workflow_service accepts custom provider and model."""
        with patch("src.app.service_factories.WorkflowService") as mock_workflow_class:
            mock_service = MagicMock()
            mock_workflow_class.return_value = mock_service

            service = get_workflow_service(provider="gemini", model="gemini-pro")

            mock_workflow_class.assert_called_once_with(provider="gemini", model="gemini-pro")
            self.assertEqual(service, mock_service)

    def test_get_workflow_service_not_cached(self):
        """Verify get_workflow_service creates new instance each time (not cached)."""
        with patch("src.app.service_factories.WorkflowService") as mock_workflow_class:
            mock_service1 = MagicMock()
            mock_service2 = MagicMock()
            mock_workflow_class.side_effect = [mock_service1, mock_service2]

            service1 = get_workflow_service()
            service2 = get_workflow_service()

            # Should create two different instances
            self.assertIsNot(service1, service2)
            self.assertEqual(mock_workflow_class.call_count, 2)

    @patch("src.app.service_factories.st")
    def test_service_factories_caching_consistency(self, mock_st):
        """Verify that cached services return same instance across multiple calls."""
        training1 = get_training_service()
        training2 = get_training_service()
        self.assertIs(training1, training2)

        inference1 = get_inference_service()
        inference2 = get_inference_service()
        self.assertIs(inference1, inference2)

        analytics1 = get_analytics_service()
        analytics2 = get_analytics_service()
        self.assertIs(analytics1, analytics2)
