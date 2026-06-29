from sqlalchemy import (
    Column, Integer, String, Date, Boolean, ForeignKey,
    Text, DECIMAL, TIMESTAMP, UniqueConstraint, Index, Enum,
    DateTime, CheckConstraint, Time,JSON
)
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func

Base = declarative_base()



class users(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    is_verified = Column(Boolean, default=False)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    last_login = Column(DateTime, default=func.now(), onupdate=func.now())

    documents = relationship("documents", back_populates="user")
    conversations = relationship("conversations", back_populates="user")

class documents(Base):
    __tablename__ = 'documents'

    id = Column(Integer, primary_key=True, index=True)
    file_name = Column(String, unique=True, index=True)
    file_type = Column(String)
    file_size = Column(Integer)
    chunks = Column(Integer)
    primary_language = Column(String)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    user_id = Column(Integer, ForeignKey('users.id'))
    user = relationship("users", back_populates="documents")


class conversations(Base):
    __tablename__ = 'conversations'

    id = Column(Integer, primary_key=True, index=True)
    conversation_uuid = Column(String,unique=True,index=True,nullable=False)
    user_id = Column(Integer, ForeignKey('users.id'))
    meta_data = Column(JSON, nullable=True)

    user = relationship("users", back_populates="conversations")
    messages = relationship("messages", back_populates="conversation")


class messages(Base):
    __tablename__ = 'messages'

    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(
        String,
        ForeignKey('conversations.conversation_uuid'),
        index=True,
        nullable=False
    )
    role = Column(String)
    content = Column(String)
    meta_data = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=func.now())
    conversation = relationship("conversations", back_populates="messages")