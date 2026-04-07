import base64
import hashlib

from cryptography.fernet import Fernet, InvalidToken
from werkzeug.security import check_password_hash, generate_password_hash


def hash_password(password: str) -> str:
    return generate_password_hash(password)


def verify_password(password_hash: str, password: str) -> bool:
    return check_password_hash(password_hash, password)


def _fernet(secret_key: str) -> Fernet:
    digest = hashlib.sha256(secret_key.encode("utf-8")).digest()
    return Fernet(base64.urlsafe_b64encode(digest))


def encrypt_secret(secret_key: str, plaintext: str) -> str:
    if not plaintext:
        return ""
    return _fernet(secret_key).encrypt(plaintext.encode("utf-8")).decode("utf-8")


def decrypt_secret(secret_key: str, ciphertext: str) -> str:
    if not ciphertext:
        return ""
    try:
        return _fernet(secret_key).decrypt(ciphertext.encode("utf-8")).decode("utf-8")
    except InvalidToken:
        return ""
