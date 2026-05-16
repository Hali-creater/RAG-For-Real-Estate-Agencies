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

# Initialize DB & Scheduler
init_db()
start_rag_scheduler()
rag_engine.initialize()

# Check for API Key
if not os.environ.get("GROQ_API_KEY"):
    st.error("⚠️ GROQ_API_KEY is not set. Please configure it in your environment variables or Streamlit secrets.")
    st.stop()

# Navigation
menu = ["Customer Chat", "Internal Assistant", "Property Management", "Lead Dashboard", "Document Upload"]
choice = st.sidebar.selectbox("Menu", menu)

db = SessionLocal()

if choice == "Customer Chat":
    st.title("💬 Customer Interaction")

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

                # Save to Leads List Button
                if st.button("💾 Save Chat to Leads List"):
                    with open("leads_list.csv", "a") as f:
                        f.write(f"{datetime.now(timezone.utc)},{lead.name},{prompt[:50]},{full_response[:100]}\n")
                    st.success("Chat saved to Leads List!")

                # Lead Capture Check (Simulated high-intent detection)
                if any(word in full_response.lower() for word in ["call", "contact", "schedule", "appointment", "visit"]):
                    st.info("💡 **Lead Intent Detected:** AI suggests capturing contact details for immediate follow-up.")
                    if st.button("🚀 Push to CRM / Webhook"):
                        st.success("Lead pushed to CRM and Agent notified via Webhook!")
            st.rerun()

elif choice == "Internal Assistant":
    st.title("💼 Agent Internal Assistant")
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

                # Internal Lead Insight Webhook Simulation
                if "client" in prompt.lower() or "deal" in prompt.lower():
                    st.toast("Insight recorded in Internal Audit Log")

elif choice == "Property Management":
    st.title("🏠 Property Listings Management")

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
    st.title("📊 Lead Intelligence Dashboard")
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
    st.title("📁 Document Knowledge Base")
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
