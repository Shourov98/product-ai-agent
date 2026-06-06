from __future__ import annotations

from dataclasses import dataclass

from fastapi import Header, HTTPException, status

from app.config import get_settings


@dataclass(slots=True)
class AuthenticatedUser:
    user_id: str
    email: str
    role: str


def get_optional_current_user(authorization: str | None = Header(default=None)) -> AuthenticatedUser | None:
    settings = get_settings()
    if not settings.auth_enabled:
        return None

    try:
        import jwt
    except ModuleNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="PyJWT is required when authentication is enabled.",
        ) from exc

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization token is required.",
        )

    token = authorization.split(" ", maxsplit=1)[1].strip()
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization token is required.",
        )

    if not settings.jwt_access_secret:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="JWT access secret is not configured.",
        )

    try:
        payload = jwt.decode(token, settings.jwt_access_secret, algorithms=["HS256"])
    except jwt.PyJWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired access token.",
        ) from exc

    user_id = str(payload.get("userId", "")).strip()
    email = str(payload.get("email", "")).strip()
    role = str(payload.get("role", "")).strip()

    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token payload is missing userId.",
        )

    _validate_user_status(user_id)
    return AuthenticatedUser(user_id=user_id, email=email, role=role or "user")


def _validate_user_status(user_id: str) -> None:
    settings = get_settings()
    if not settings.mongodb_enabled or not settings.mongodb_uri:
        return

    try:
        from pymongo import MongoClient
    except ModuleNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="pymongo is required when MongoDB-backed auth validation is enabled.",
        ) from exc

    client = MongoClient(settings.mongodb_uri)
    try:
        database = client[settings.mongodb_db_name]
        user = database[settings.mongodb_users_collection].find_one({"_id": _coerce_object_id(user_id)})
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found for this token.",
            )
        if str(user.get("status", "")).lower() == "blocked":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Your account is blocked.",
            )
    finally:
        client.close()


def _coerce_object_id(value: str):
    try:
        from bson import ObjectId

        return ObjectId(value)
    except Exception:
        return value
