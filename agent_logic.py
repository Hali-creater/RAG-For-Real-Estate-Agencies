import os
import json
from typing import List, Optional
from langchain_groq import ChatGroq
from rag_engine import rag_engine
from models import Lead, ChatHistory, InternalQuery
from sqlalchemy.orm import Session
from datetime import datetime, timezone

def get_llm():
    if not os.environ.get("GROQ_API_KEY"):
        return None
    return ChatGroq(model="llama-3.3-70b-specdec")

MASTER_PROMPT = """
MASTER RAG AGENT PROMPT (REAL ESTATE AI AGENT)

You are a precise Real Estate Assistant.
1. Only answer using the provided context. If the answer is not in the context, say: "I cannot find that detail in the current files. Let me log this for the agent to check."
2. NEVER mention or analyze neighborhood safety, crime rates, race, or religious demographics (this violates Fair Housing laws).
3. At the end of EVERY answer, add a section called 'Source Document:' and name the exact file where you found the information.

Your primary goal is to respond instantly to inquiries, recommend the most relevant properties, qualify leads, and assist agents in closing deals efficiently.

When interacting with potential buyers or investors, you must answer all property-related questions clearly and accurately using available data such as property listings (price, location, size, features), FAQs, project brochures (PDFs), and CRM data. Based on the user’s requirements, intelligently recommend matching properties and guide the conversation by asking relevant follow-up questions such as budget, preferred location, property type, and financing capability. Your objective is to identify serious buyers and filter out low-intent inquiries while capturing key lead information for the business.

You should also act as an automated follow-up and marketing assistant. If a user shows interest but does not take action, continue engagement by sending reminders, suggesting similar properties, sharing updates like price changes, and nurturing the lead until conversion. Ensure no lead is ignored or lost due to delayed responses.

In addition, you function as an internal AI assistant for real estate teams. You can search and retrieve information from internal documents, contracts, previous deals, and CRM records. You should assist agents by answering internal queries, such as identifying clients based on preferences, retrieving past interactions, and providing insights that help close deals faster.

You must support multilingual communication to handle international clients. Detect the user’s language and respond accordingly in English, Arabic, Russian, Chinese, or other relevant languages while maintaining clarity and professionalism. This is especially important for markets with foreign investors.

You also provide investment guidance when required. Analyze available data to suggest the best areas for investment, compare properties, and estimate potential returns such as rental yield or ROI. Your responses should be data-driven, realistic, and helpful for decision-making, while clearly stating when estimates are approximate or dependent on market conditions.

Your communication style should always be professional, concise, and persuasive. Focus on delivering value, guiding the user toward a decision, and increasing the likelihood of conversion. Avoid unnecessary information and prioritize actionable insights.

Your ultimate purpose is to help the real estate business save time, improve efficiency, increase lead conversion rates, and generate more revenue through intelligent automation.
"""

class MasterRAGAgent:
    def __init__(self, db: Session):
        self.db = db

    def get_chat_history(self, lead_id: int):
        history = self.db.query(ChatHistory).filter(ChatHistory.lead_id == lead_id).order_by(ChatHistory.timestamp.asc()).all()
        return [{"role": h.role, "content": h.content} for h in history]

    def save_message(self, lead_id: int, role: str, content: str):
        message = ChatHistory(lead_id=lead_id, role=role, content=content)
        self.db.add(message)
        self.db.commit()

    def stream_customer_query(self, lead_id: int, query: str):
        llm = get_llm()
        if not llm:
            yield "Error: Groq API key not configured. Please set GROQ_API_KEY."
            return

        # 1. Retrieve relevant context from RAG
        try:
            context_docs = rag_engine.query(query)
            context_text = ""
            for doc in context_docs:
                src = doc.metadata.get("source", "Unknown")
                context_text += f"[FROM SOURCE: {src}]\n{doc.page_content}\n\n"
        except Exception as e:
            print(f"RAG Retrieval Error: {e}")
            context_text = "No additional context available."

        # 2. Get chat history
        history = self.get_chat_history(lead_id)

        # 3. Prepare messages for Groq
        messages = [("system", MASTER_PROMPT + f"\n\nCONTEXT FROM DATA SOURCE:\n{context_text}")]
        for h in history:
            messages.append((h["role"], h["content"]))
        messages.append(("user", query))

        # 4. Stream response from Groq
        full_answer = ""
        for chunk in llm.stream(messages):
            content = chunk.content
            full_answer += content
            yield content

        # 5. Save history
        self.save_message(lead_id, "user", query)
        self.save_message(lead_id, "assistant", full_answer)

    def stream_internal_query(self, agent_name: str, query: str):
        llm = get_llm()
        if not llm:
            yield "Error: Groq API key not configured. Please set GROQ_API_KEY."
            return

        # Similar logic but focused on internal team needs
        try:
            context_docs = rag_engine.query(query)
            context_text = ""
            for doc in context_docs:
                src = doc.metadata.get("source", "Unknown")
                context_text += f"[FROM SOURCE: {src}]\n{doc.page_content}\n\n"
        except Exception as e:
            print(f"RAG Retrieval Error: {e}")
            context_text = "No additional context available."

        messages = [
            ("system", MASTER_PROMPT + f"\n\nYou are currently assisting a team member: {agent_name}. Focus on internal data retrieval and deal-closing insights.\n\nCONTEXT FROM INTERNAL DATA:\n{context_text}"),
            ("user", query)
        ]

        full_answer = ""
        for chunk in llm.stream(messages):
            content = chunk.content
            full_answer += content
            yield content

        # Save internal query
        internal_rec = InternalQuery(agent_name=agent_name, query=query, response=full_answer)
        self.db.add(internal_rec)
        self.db.commit()

    def handle_customer_query(self, lead_id: int, query: str):
        # Compatibility wrapper
        return "".join(list(self.stream_customer_query(lead_id, query)))

    def handle_internal_query(self, agent_name: str, query: str):
        # Compatibility wrapper
        return "".join(list(self.stream_internal_query(agent_name, query)))
