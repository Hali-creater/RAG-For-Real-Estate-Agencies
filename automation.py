import os
from openai import OpenAI
from communication import send_multilingual_followup

client = OpenAI()

async def generate_and_send_followup(db, lead_id: int):
    from models import Lead, ChatHistory
    lead = db.query(Lead).filter(Lead.id == lead_id).first()
    if not lead:
        return

    # Get recent chat history to personalize follow-up
    history = db.query(ChatHistory).filter(ChatHistory.lead_id == lead_id).order_by(ChatHistory.timestamp.desc()).limit(5).all()
    history_text = "\n".join([f"{h.role}: {h.content}" for h in reversed(history)])

    prompt = f"""
    Based on the following chat history with lead {lead.name}, generate a professional, persuasive, and helpful follow-up message.
    The goal is to nurture the lead and move them closer to a decision.

    CHAT HISTORY:
    {history_text}

    LEAD DETAILS:
    Language: {lead.language}
    Budget: {lead.budget}
    Location Pref: {lead.preferred_location}

    If the history shows interest in a specific property or area, mention it.
    The response should be in the lead's preferred language ({lead.language}).
    Keep it concise and actionable.
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    followup_content = response.choices[0].message.content

    await send_multilingual_followup(lead_id, followup_content, lead.language)
