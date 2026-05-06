from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from database import get_db, init_db
from agent_logic import MasterRAGAgent
from scheduler import start_rag_scheduler
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(title="Master RAG Real Estate Agent API")

init_db()
start_rag_scheduler()

class QueryRequest(BaseModel):
    lead_id: int
    query: str

class InternalQueryRequest(BaseModel):
    agent_name: str
    query: str

class QueryResponse(BaseModel):
    response: str

@app.get("/")
def read_root():
    return {"message": "Master RAG Agent API is live"}

@app.post("/customer-query", response_model=QueryResponse)
def customer_query(req: QueryRequest, db: Session = Depends(get_db)):
    agent = MasterRAGAgent(db)
    try:
        response = agent.handle_customer_query(req.lead_id, req.query)
        return QueryResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/internal-query", response_model=QueryResponse)
def internal_query(req: InternalQueryRequest, db: Session = Depends(get_db)):
    agent = MasterRAGAgent(db)
    try:
        response = agent.handle_internal_query(req.agent_name, req.query)
        return QueryResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
