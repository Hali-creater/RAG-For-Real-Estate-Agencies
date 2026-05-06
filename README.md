# Master RAG Real Estate Agent

An advanced, global AI assistant designed specifically for real estate businesses to maximize lead conversion, automate communication, and support both customers and internal teams using Retrieval-Augmented Generation (RAG).

## 🚀 Key Features

*   **Global RAG Engine**: Intelligently indexes property listings, project brochures (PDFs), FAQs, and internal documents to provide data-driven responses.
*   **Multilingual Support**: Automatically detects and responds in English, Arabic, Russian, Chinese, and more, making it ideal for international investors and markets.
*   **Customer Interaction**: Instantly answers property-related questions, recommends relevant listings, and qualifies leads based on budget and preferences.
*   **Internal Team Assistant**: Helps real estate teams retrieve deal insights, past interactions, and contract details from internal records.
*   **Investment Guidance**: Provides data-driven analysis on rental yields, ROI, and property comparisons across different markets.
*   **Automated Nurturing**: Background follow-up engine that continues engagement with leads using personalized, multilingual messages.

## 🛠️ Tech Stack

*   **Backend**: Python, FastAPI
*   **Frontend**: Streamlit
*   **Database**: SQLite (SQLAlchemy ORM)
*   **Intelligence**: OpenAI (GPT-4o, Embeddings), LangChain, FAISS
*   **Automation**: APScheduler

## 📂 Project Structure

```text
.
├── app.py              # Streamlit Application (Main UI)
├── main.py             # FastAPI Backend & API
├── agent_logic.py      # Master RAG Agent Core Logic
├── rag_engine.py       # Vector Store & Retrieval Logic
├── models.py           # Database Models
├── database.py         # Database Configuration
├── automation.py       # AI-driven Follow-up Logic
├── communication.py    # Multilingual Messaging Stubs
├── telegram_bot.py     # Agent Alert System
└── scheduler.py        # Background Task Automation
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API Keys
Ensure your `OPENAI_API_KEY` is set in your environment.

### 3. Run the Application

#### Streamlit Dashboard
```bash
streamlit run app.py
```

#### FastAPI API
```bash
uvicorn main:app --reload
```

## 🌐 Deployment

### Streamlit Cloud
1. Push this code to a GitHub repository.
2. Connect to [Streamlit Cloud](https://share.streamlit.io/).
3. In settings, ensure the **Main file path** is set to `app.py`.
4. Add your OpenAI and other secrets in the "Settings > Secrets" section.

### Docker / VPS
The application can be containerized or run on any VPS supporting Python, providing a robust backend for property data and lead management.
