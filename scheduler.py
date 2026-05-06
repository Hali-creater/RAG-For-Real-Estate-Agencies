from apscheduler.schedulers.background import BackgroundScheduler
from database import SessionLocal
from automation import generate_and_send_followup
import asyncio

scheduler = BackgroundScheduler()

def check_for_followups():
    db = SessionLocal()
    # Logic to identify leads who need follow-up (e.g., no interaction for 3 days)
    # This is a simplified version
    from models import Lead
    from datetime import datetime, timezone, timedelta

    threshold = datetime.now(timezone.utc) - timedelta(days=3)
    leads_to_follow = db.query(Lead).filter(Lead.last_interaction < threshold).all()

    for lead in leads_to_follow:
        asyncio.run(generate_and_send_followup(db, lead.id))
        lead.last_interaction = datetime.now(timezone.utc)
        db.commit()

    db.close()

def start_rag_scheduler():
    if not scheduler.running:
        scheduler.add_job(check_for_followups, 'interval', hours=24)
        scheduler.start()
