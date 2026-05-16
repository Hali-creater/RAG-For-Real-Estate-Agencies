import streamlit as st
import os
import pandas as pd
from database import SessionLocal, init_db
from models import Lead, Property, Document as DBDocument, ChatHistory
from agent_logic import MasterRAGAgent
from rag_engine import rag_engine
from scheduler import start_rag_scheduler
from datetime import datetime, timezone

# Page Config
st.set_page_config(page_title="Master RAG Real Estate Agent", layout="wide", page_icon="🏠")

# Custom CSS for Styling
st.markdown("""
<style>
    /* Main Background and Typography */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    .stApp {
        background: linear-gradient(135deg, #f8f9fc 0%, #f1f4f8 100%);
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
    }
    ::-webkit-scrollbar-thumb {
        background: #d0d5dd;
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #98a2b3;
    }

    /* Fade-in Animation */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Sidebar Styling - Light/White Theme */
    section[data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #f2f4f7;
    }
    section[data-testid="stSidebar"] .stMarkdown h1 {
        color: #101828;
        font-size: 1.5rem;
        font-weight: 700;
        padding-bottom: 1rem;
    }
    .stSidebar [data-testid="stMarkdownContainer"] p {
        color: #475467;
    }

    /* Radio Button Navigation in Sidebar */
    div[data-testid="stSidebar"] div[role="radiogroup"] > label {
        padding: 10px 15px;
        border-radius: 10px;
        transition: all 0.3s ease;
        border: 1px solid transparent;
        margin-bottom: 4px;
    }

    div[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
        background-color: #f0f2f6;
    }

    div[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
        background-color: #f0f7ff !important;
        border: 1px solid #cce4ff !important;
    }

    div[data-testid="stSidebar"] div[role="radiogroup"] label [data-testid="stMarkdownContainer"] p {
        font-size: 1.3rem !important;
        font-weight: 600 !important;
        color: #101828 !important;
        line-height: 1.4 !important;
    }

    /* Chat Message Styling with Glassmorphism */
    .stChatMessage {
        border-radius: 24px;
        padding: 1.8rem;
        margin-bottom: 1.8rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        animation: fadeIn 0.5s ease-out;
        border: 1px solid rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(8px);
    }
    [data-testid="stChatMessage"]:nth-child(even) {
        background-color: rgba(255, 255, 255, 0.7);
    }
    [data-testid="stChatMessage"]:nth-child(odd) {
        background-color: rgba(249, 250, 251, 0.7);
    }

    /* Button Styling - Modern Flat Design */
    .stButton>button {
        border-radius: 14px;
        border: 1px solid #e4e7ec;
        background: white;
        color: #344054;
        font-weight: 600;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        padding: 0.7rem 1.5rem;
        box-shadow: 0 1px 2px rgba(16, 24, 40, 0.05);
    }
    .stButton>button:hover {
        background: #fcfcfd;
        border-color: #1570ef;
        color: #1570ef;
        transform: translateY(-1px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }

    /* Primary Action Buttons */
    div.stColumns div.stButton>button {
        width: 100%;
        background: #ffffff;
        border: 1px solid #eaecf0;
    }
    div.stColumns div.stButton>button:hover {
        background: #fefefe;
        border-color: #d0d5dd;
    }

    /* Table Styling */
    [data-testid="stDataFrame"] {
        background: white;
        border-radius: 16px;
        border: 1px solid #eaecf0;
        overflow: hidden;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }

    /* Input Field Styling */
    .stTextInput>div>div>input, .stSelectbox>div>div>div, .stTextArea>div>div>textarea {
        border-radius: 12px !important;
        border: 1px solid #d0d5dd !important;
        padding: 0.6rem 1rem !important;
        background-color: white !important;
        transition: all 0.2s ease !important;
    }
    .stTextInput>div>div>input:focus, .stSelectbox>div>div>div:focus, .stTextArea>div>div>textarea:focus {
        border-color: #1570ef !important;
        box-shadow: 0 0 0 4px rgba(21, 112, 239, 0.1) !important;
    }

    /* Status Info Styling */
    .stAlert {
        border-radius: 12px;
        border: 1px solid #eaecf0;
        background: white;
    }

    /* Custom Header Bar */
    .custom-header {
        background: linear-gradient(to right, #ffffff, #f9fafb);
        padding: 2.5rem;
        border-radius: 24px;
        border: 1px solid #eaecf0;
        margin-bottom: 3rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05);
    }
    .custom-header h1 {
        font-size: 1.8rem !important;
        font-weight: 700;
        color: #101828;
        margin: 0;
    }
    .custom-header span {
        color: #667085;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Initialize DB & Scheduler
init_db()
start_rag_scheduler()
rag_engine.initialize()

# Check for API Key
if not os.environ.get("GROQ_API_KEY"):
    st.error("⚠️ GROQ_API_KEY is not set. Please configure it in your environment variables or Streamlit secrets.")
    st.stop()

# Navigation
st.sidebar.title("🏢 Agent Command Center")
st.sidebar.markdown("---")
menu = {
    "💬\u00A0\u00A0Customer Chat": "Customer Chat",
    "💼\u00A0\u00A0Internal Assistant": "Internal Assistant",
    "🏠\u00A0\u00A0Property Management": "Property Management",
    "📊\u00A0\u00A0Lead Dashboard": "Lead Dashboard",
    "📁\u00A0\u00A0Document Upload": "Document Upload"
}
st.sidebar.subheader("Main Navigation")
choice_label = st.sidebar.radio("Navigation", list(menu.keys()), label_visibility="collapsed")
choice = menu[choice_label]

st.sidebar.divider()
st.sidebar.info("Select a module above to begin. All responses are powered by high-speed Groq inference.")

db = SessionLocal()

if choice == "Customer Chat":
    st.markdown('<div class="custom-header"><h1>💬 Customer Interaction Center</h1><span>Powered by Groq Intelligence</span></div>', unsafe_allow_html=True)

    # Select or create lead
    leads = db.query(Lead).all()
    lead_names = [f"{l.id}: {l.name}" for l in leads]
    selected_lead_str = st.selectbox("Select Lead", ["New Lead"] + lead_names)

    lead_id = None
    if selected_lead_str == "New Lead":
        with st.form("new_lead_form"):
            name = st.text_input("Name")
            email = st.text_input("Email")
            phone = st.text_input("Phone")
            submitted = st.form_submit_button("Create Lead")
            if submitted:
                new_lead = Lead(name=name, email=email, phone=phone)
                db.add(new_lead)
                db.commit()
                db.refresh(new_lead)
                st.success(f"Lead created with ID {new_lead.id}")
                st.rerun()
    else:
        lead_id = int(selected_lead_str.split(":")[0])
        lead = db.query(Lead).filter(Lead.id == lead_id).first()
        st.write(f"Chatting with **{lead.name}**")

        agent = MasterRAGAgent(db)

        # Display chat history
        history = agent.get_chat_history(lead_id)
        for msg in history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        # Quick Action Buttons
        col_btn1, col_btn2, col_btn3 = st.columns(3)
        q_prompt = None
        if col_btn1.button("✨ Generate Client Summary"):
            q_prompt = "Please turn the property details and disclosures into a 5-point bullet list summary for a client."
        if col_btn2.button("📧 Draft Follow-Up Email"):
            q_prompt = "Write a professional email pitch to this lead using the property details from our database."
        if col_btn3.button("🛡️ Run Compliance Check"):
            q_prompt = "Scan the conversation and property data to ensure there are no legal issues or missing disclosure information."

        # Chat input
        prompt = st.chat_input("Ask about properties or investment guidance...")
        if q_prompt:
            prompt = q_prompt

        if prompt:
            with st.chat_message("user"):
                st.write(prompt)

            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                full_response = ""
                for chunk in agent.stream_customer_query(lead_id, prompt):
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")
                response_placeholder.markdown(full_response)

                # Store in session state for persistence
                st.session_state.last_response = full_response
                st.session_state.last_prompt = prompt
            st.rerun()

        # Interaction Area (persists after response)
        if "last_response" in st.session_state and lead_id:
            st.divider()
            col_save, col_push = st.columns(2)
            if col_save.button("💾 Save Chat to Leads List"):
                with open("leads_list.csv", "a") as f:
                    f.write(f"{datetime.now(timezone.utc)},{lead.name},{st.session_state.last_prompt[:50]},{st.session_state.last_response[:100]}\n")
                st.success("Chat saved to Leads List!")

            # Lead Capture Check (Simulated high-intent detection)
            if any(word in st.session_state.last_response.lower() for word in ["call", "contact", "schedule", "appointment", "visit"]):
                st.info("💡 **Lead Intent Detected:** AI suggests capturing contact details for immediate follow-up.")
                if col_push.button("🚀 Push to CRM / Webhook"):
                    st.success("Lead pushed to CRM and Agent notified via Webhook!")

elif choice == "Internal Assistant":
    st.markdown('<div class="custom-header"><h1>💼 Agent Intelligence Hub</h1><span>Internal Data Analytics</span></div>', unsafe_allow_html=True)
    agent_name = st.text_input("Your Name", "Agent Smith")

    if "internal_history" not in st.session_state:
        st.session_state.internal_history = []

    for msg in st.session_state.internal_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if prompt := st.chat_input("Search internal docs, client preferences, or deal info..."):
        st.session_state.internal_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        agent = MasterRAGAgent(db)
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            for chunk in agent.stream_internal_query(agent_name, prompt):
                full_response += chunk
                response_placeholder.markdown(full_response + "▌")
            response_placeholder.markdown(full_response)
            st.session_state.internal_history.append({"role": "assistant", "content": full_response})

            # Internal Lead Insight Webhook Simulation
            if "client" in prompt.lower() or "deal" in prompt.lower():
                st.toast("Insight recorded in Internal Audit Log")

elif choice == "Property Management":
    st.markdown('<div class="custom-header"><h1>🏠 Global Property Portfolio</h1><span>Inventory & Distribution</span></div>', unsafe_allow_html=True)

    with st.expander("Add New Property"):
        with st.form("add_property"):
            title = st.text_input("Title")
            desc = st.text_area("Description")
            price = st.number_input("Price", min_value=0.0)
            location = st.text_input("Location")
            size = st.text_input("Size (sqft/sqm)")
            p_type = st.selectbox("Type", ["Apartment", "Villa", "Townhouse", "Land", "Commercial"])
            features = st.text_area("Features (comma separated)")

            if st.form_submit_button("Save Property"):
                new_p = Property(title=title, description=desc, price=price, location=location, size=size, property_type=p_type, features=features)
                db.add(new_p)
                db.commit()
                st.success("Property added!")
                rag_engine.add_documents_from_db(db) # Refresh index

    st.subheader("Existing Properties")
    properties = db.query(Property).all()
    if properties:
        df = pd.DataFrame([{
            "ID": p.id, "Title": p.title, "Price": f"{p.price:,.2f}", "Location": p.location, "Type": p.property_type
        } for p in properties])
        st.dataframe(df, use_container_width=True)

elif choice == "Lead Dashboard":
    st.markdown('<div class="custom-header"><h1>📊 Lead Analytics & Tracking</h1><span>Conversion Metrics</span></div>', unsafe_allow_html=True)
    leads = db.query(Lead).all()
    if leads:
        data = []
        for l in leads:
            data.append({
                "ID": l.id, "Name": l.name, "Email": l.email, "Phone": l.phone,
                "Lang": l.language, "Created": l.created_at.strftime("%Y-%m-%d")
            })
        st.dataframe(pd.DataFrame(data), use_container_width=True)
    else:
        st.info("No leads captured yet.")

elif choice == "Document Upload":
    st.markdown('<div class="custom-header"><h1>📁 Knowledge Base Engine</h1><span>Document Intelligence</span></div>', unsafe_allow_html=True)
    st.write("Upload PDFs or TXT files (Brochures, FAQs, Contracts, Market Reports)")

    doc_type = st.selectbox("Document Type", ["Brochure", "FAQ", "Contract", "Market Report", "Other"])
    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "txt"])

    if uploaded_file is not None:
        save_path = os.path.join("uploads", uploaded_file.name)
        os.makedirs("uploads", exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if st.button("Index Document"):
            new_doc = DBDocument(title=uploaded_file.name, file_path=save_path, doc_type=doc_type)
            db.add(new_doc)
            db.commit()
            st.success(f"File {uploaded_file.name} saved and indexing...")
            rag_engine.add_documents_from_db(db)
            st.success("Indexing complete!")

db.close()
