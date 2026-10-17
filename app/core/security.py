from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID

from jose import JWTError, jwt
from passlib.context import CryptContext

from app.core.config import settings

ALGORITHM = "RS256"
password_context = CryptContext(schemes=["bcrypt_sha256"], deprecated="auto")


def _read_key(path: str, key_type: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(
            f"JWT {key_type} key could not be read from {path!r}; configure "
            f"JWT_{key_type.upper()}_KEY_PATH"
        ) from exc


def hash_password(password: str) -> str:
    return password_context.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    return password_context.verify(password, password_hash)


def create_access_token(subject: UUID) -> str:
    expires_at = datetime.now(UTC) + timedelta(minutes=settings.access_token_expire_minutes)
    return jwt.encode(
        {"sub": str(subject), "exp": expires_at},
        _read_key(settings.jwt_private_key_path, "private"),
        algorithm=ALGORITHM,
    )


def decode_access_token(token: str) -> UUID | None:
    try:
        subject = jwt.decode(
            token,
            _read_key(settings.jwt_public_key_path, "public"),
            algorithms=[ALGORITHM],
        ).get("sub")
        return UUID(subject) if isinstance(subject, str) else None
    except (JWTError, ValueError):
        return None
