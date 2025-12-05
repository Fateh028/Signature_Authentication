from passlib.context import CryptContext

# NOTE:
# There is a known compatibility issue between some Windows Python builds,
# `passlib`, and the `bcrypt` backend which is causing runtime errors in
# your environment. To keep things simple and robust, we switch to the
# built‑in `pbkdf2_sha256` scheme from passlib, which is a strong and
# widely used password hash and does NOT depend on the external `bcrypt`
# library.

pwd_context = CryptContext(
    schemes=["pbkdf2_sha256"],
    deprecated="auto",
)


def hash_password(password: str) -> str:
    """
    Hash a plain-text password using PBKDF2-SHA256.
    """
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed: str) -> bool:
    """
    Verify a candidate password against a stored hash.
    """
    return pwd_context.verify(plain_password, hashed)

