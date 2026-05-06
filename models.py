from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime, timezone

Base = declarative_base()

class Property(Base):
    __tablename__ = "properties"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    description = Column(Text)
    price = Column(Float)
    location = Column(String)
    size = Column(String)
    features = Column(Text) # JSON string or comma-separated
    property_type = Column(String) # Villa, Apartment, etc.
    status = Column(String, default="Available") # Available, Sold
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

class Lead(Base):
    __tablename__ = "leads"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    email = Column(String)
    phone = Column(String)
    budget = Column(Float)
    preferred_location = Column(String)
    property_type_pref = Column(String)
    financing_capability = Column(String)
    intent_score = Column(Integer, default=0)
    language = Column(String, default="English")
    last_interaction = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    chat_history = relationship("ChatHistory", back_populates="lead")

class ChatHistory(Base):
    __tablename__ = "chat_history"
    id = Column(Integer, primary_key=True, index=True)
    lead_id = Column(Integer, ForeignKey("leads.id"))
    role = Column(String) # user, assistant
    content = Column(Text)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    lead = relationship("Lead", back_populates="chat_history")

class Document(Base):
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String)
    file_path = Column(String)
    doc_type = Column(String) # FAQ, Brochure, Contract, Deal
    content_summary = Column(Text)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

class InternalQuery(Base):
    __tablename__ = "internal_queries"
    id = Column(Integer, primary_key=True, index=True)
    agent_name = Column(String)
    query = Column(Text)
    response = Column(Text)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
