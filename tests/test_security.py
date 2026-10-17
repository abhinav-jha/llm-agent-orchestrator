from uuid import uuid4

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from jose import jwt
import pytest

from app.core.config import settings
from app.core.security import create_access_token, decode_access_token, hash_password, verify_password


@pytest.fixture
def rsa_key_paths(tmp_path, monkeypatch) -> None:
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private_path = tmp_path / "private.pem"
    public_path = tmp_path / "public.pem"
    private_path.write_bytes(
        private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    public_path.write_bytes(
        private_key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    monkeypatch.setattr(settings, "jwt_private_key_path", str(private_path))
    monkeypatch.setattr(settings, "jwt_public_key_path", str(public_path))


def test_password_hash_verifies_only_original_password() -> None:
    password_hash = hash_password("correct horse battery staple")

    assert verify_password("correct horse battery staple", password_hash)
    assert not verify_password("incorrect password", password_hash)


def test_access_token_round_trips_user_id(rsa_key_paths) -> None:
    user_id = uuid4()
    token = create_access_token(user_id)

    assert jwt.get_unverified_header(token)["alg"] == "RS256"
    assert decode_access_token(token) == user_id


def test_invalid_access_token_is_rejected(rsa_key_paths) -> None:
    assert decode_access_token("not-a-jwt") is None
