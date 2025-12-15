"""FastAPI dependencies for authentication and other shared functionality."""

from typing import Optional

from fastapi import Header, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.api.exceptions import AuthenticationError
from src.utils.env_loader import get_env_var
from src.utils.logger import get_logger

logger = get_logger(__name__)

security = HTTPBearer(auto_error=False)


async def verify_api_key(
    authorization: Optional[HTTPAuthorizationCredentials] = Security(security),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> str:
    """Verify API key from header. Args: authorization (Optional[HTTPAuthorizationCredentials]): Bearer token. x_api_key (Optional[str]): X-API-Key header. Returns: str: User ID or API key identifier. Raises: AuthenticationError: If authentication fails."""
    api_key_from_env = get_env_var("API_KEY")

    provided_key = None
    if authorization:
        provided_key = authorization.credentials
    elif x_api_key:
        provided_key = x_api_key

    if not api_key_from_env:
        logger.warning("API_KEY not set in environment - allowing access for development")
        return provided_key or "default_user"

    if not provided_key:
        raise AuthenticationError(
            "API key required. Provide via Authorization: Bearer <key> or X-API-Key header"
        )

    if provided_key != api_key_from_env:
        raise AuthenticationError("Invalid API key")

    return str(provided_key)


async def get_current_user(api_key: str = Security(verify_api_key)) -> str:
    """Get current user from API key. Args: api_key (str): Verified API key. Returns: str: User identifier."""
    api_key_from_env = get_env_var("API_KEY")
    if not api_key_from_env:
        return api_key
    return api_key
