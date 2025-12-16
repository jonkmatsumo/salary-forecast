"""Custom exceptions for API endpoints."""

from typing import Any, Dict, Optional


class APIException(Exception):
    """Base exception for API errors."""

    def __init__(
        self, code: str, message: str, status_code: int, details: Optional[Dict[str, Any]] = None
    ):
        """Initialize API exception.

        Args:
            code (str): Error code.
            message (str): Error message.
            status_code (int): HTTP status code.
            details (Optional[Dict[str, Any]]): Additional error details.
        """
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)


class ModelNotFoundError(APIException):
    """Raised when a model cannot be found."""

    def __init__(self, run_id: str, details: Optional[Dict[str, Any]] = None):
        """Initialize model not found error.

        Args:
            run_id (str): Model run ID.
            details (Optional[Dict[str, Any]]): Additional details.
        """
        super().__init__(
            code="MODEL_NOT_FOUND",
            message=f"Model with run_id '{run_id}' not found",
            status_code=404,
            details=details or {"run_id": run_id},
        )


class InvalidInputError(APIException):
    """Raised when input validation fails."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """Initialize invalid input error.

        Args:
            message (str): Error message.
            details (Optional[Dict[str, Any]]): Additional details.
        """
        super().__init__(
            code="VALIDATION_ERROR",
            message=message,
            status_code=400,
            details=details or {},
        )


class TrainingJobNotFoundError(APIException):
    """Raised when a training job cannot be found."""

    def __init__(self, job_id: str, details: Optional[Dict[str, Any]] = None):
        """Initialize training job not found error.

        Args:
            job_id (str): Job ID.
            details (Optional[Dict[str, Any]]): Additional details.
        """
        super().__init__(
            code="TRAINING_JOB_NOT_FOUND",
            message=f"Training job with id '{job_id}' not found",
            status_code=404,
            details=details or {"job_id": job_id},
        )


class WorkflowNotFoundError(APIException):
    """Raised when a workflow cannot be found."""

    def __init__(self, workflow_id: str, details: Optional[Dict[str, Any]] = None):
        """Initialize workflow not found error.

        Args:
            workflow_id (str): Workflow ID.
            details (Optional[Dict[str, Any]]): Additional details.
        """
        super().__init__(
            code="WORKFLOW_NOT_FOUND",
            message=f"Workflow with id '{workflow_id}' not found",
            status_code=404,
            details=details or {"workflow_id": workflow_id},
        )


class WorkflowInvalidStateError(APIException):
    """Raised when workflow is in wrong state for operation."""

    def __init__(self, workflow_id: str, current_phase: str, required_phase: str):
        """Initialize workflow invalid state error.

        Args:
            workflow_id (str): Workflow ID.
            current_phase (str): Current phase.
            required_phase (str): Required phase.
        """
        super().__init__(
            code="WORKFLOW_INVALID_STATE",
            message=f"Workflow {workflow_id} is in phase '{current_phase}', but operation requires '{required_phase}'",
            status_code=400,
            details={
                "workflow_id": workflow_id,
                "current_phase": current_phase,
                "required_phase": required_phase,
            },
        )


class AuthenticationError(APIException):
    """Raised when authentication fails."""

    def __init__(self, message: str = "Authentication required"):
        """Initialize authentication error.

        Args:
            message (str): Error message.
        """
        super().__init__(
            code="AUTHENTICATION_REQUIRED",
            message=message,
            status_code=401,
        )


class AuthorizationError(APIException):
    """Raised when authorization fails."""

    def __init__(self, message: str = "Insufficient permissions"):
        """Initialize authorization error.

        Args:
            message (str): Error message.
        """
        super().__init__(
            code="AUTHORIZATION_FAILED",
            message=message,
            status_code=403,
        )


class RateLimitError(APIException):
    """Raised when rate limit is exceeded."""

    def __init__(self, message: str = "Rate limit exceeded"):
        """Initialize rate limit error.

        Args:
            message (str): Error message.
        """
        super().__init__(
            code="RATE_LIMIT_EXCEEDED",
            message=message,
            status_code=429,
        )
