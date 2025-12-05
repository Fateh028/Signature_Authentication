from sqlalchemy import Column, Integer, String, DateTime, Text, UniqueConstraint
from sqlalchemy.sql import func

from database import Base


class User(Base):
    """
    User model storing username, password hash and signature template.

    Signature template is stored as a JSON-encoded list of floats
    representing either a deep embedding or a flattened preprocessed image.
    """

    __tablename__ = "users"
    __table_args__ = (UniqueConstraint("name", name="uq_user_name"),)

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(128), nullable=False, index=True)
    password_hash = Column(String(256), nullable=False)
    signature_embedding = Column(Text, nullable=False)  # JSON-encoded vector
    created_at = Column(DateTime(timezone=True), server_default=func.now())


