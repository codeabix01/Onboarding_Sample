import logging
import re
import json
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Logging setup
logging.basicConfig(level=logging.DEBUG)

# Initialize FastAPI app
app = FastAPI()

# Load intent configuration (this could also come from a DB)
with open("intent_config.json", "r") as f:
    intent_config = json.load(f)

# Flatten examples and map back to intents
intent_examples = [(intent['intent'], ex) for intent in intent_config for ex in intent['examples']]
intent_phrases = [ex[1] for ex in intent_examples]
intent_labels = [ex[0] for ex in intent_examples]

# Load sentence-transformers model with error handling
try:
    model = SentenceTransformer('./local-model')
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model: {str(e)}")
    raise

# Precompute intent embeddings
intent_embeddings = model.encode(intent_phrases)

# Pydantic model for request
class UserQuery(BaseModel):
    message: str

# Load client data (this would be your JSON data)
with open("clients_data.json", "r") as f:
    client_data = json.load(f)

# Extract company name from message
def extract_company(text: str):
    try:
        companies = set()
        # Log the first few records for debugging
        logging.debug(f"Sample data: {client_data[:3]}")  # Log first 3 records
        for doc in client_data:
            if "company" in doc:
                companies.add(doc["company"])
        logging.debug(f"Extracted companies: {companies}")
        for company in companies:
            if company.lower() in text.lower():
                logging.debug(f"Company matched: {company}")
                return company
    except Exception as e:
        logging.error(f"Error extracting company: {str(e)}")
    return None

# Extract WCIS ID from the user query
def extract_wcis_id(user_text: str):
    try:
        wcis_id_match = re.search(r"\b\d{6,}\b", user_text)
        if wcis_id_match:
            wcis_id = wcis_id_match.group(0)
            logging.debug(f"WCIS ID matched: {wcis_id}")
            return wcis_id
        else:
            logging.debug("No WCIS ID matched.")
    except Exception as e:
        logging.error(f"Error extracting WCIS ID: {str(e)}")
    return None

# Intent-specific handlers
def handle_onboarding_status(doc):
    return f"{doc.get('company', 'Unknown')} onboarding is {doc.get('status', 'unknown')} complete."

def handle_pending_steps(doc):
    pending = [step for step, status in doc.get("steps", {}).items() if status != "complete"]
    return f"Pending steps for {doc.get('company', 'Unknown')}: {', '.join(pending) if pending else 'None'}"

def handle_step_status(doc, user_text):
    for step in ["KYC", "AccountOpening", "LegalEntity"]:
        if step.lower() in user_text.lower():
            return f"{step} status for {doc.get('company', 'Unknown')}: {doc.get('steps', {}).get(step, 'unknown')}"
    return "Step not recognized."

def handle_who_is_customer(doc, wcis_id):
    return f"WCIS ID {wcis_id} belongs to {doc.get('company', 'Unknown')}"

intent_handlers = {
    "onboarding_status": handle_onboarding_status,
    "pending_steps": handle_pending_steps,
    "step_status": handle_step_status,
    "who_is_customer": handle_who_is_customer
}

# Main API handler
@app.post("/query")
def handle_query(q: UserQuery):
    try:
        user_text = q.message
        logging.debug(f"User query: {user_text}")

        # Extract company and WCIS ID
        company = extract_company(user_text)
        wcis_id = extract_wcis_id(user_text)

        logging.debug(f"Extracted WCIS ID: {wcis_id}, Company: {company}")

        # Find the matching document from the in-memory client data
        doc = next((doc for doc in client_data if (not wcis_id or doc["wcis_id"] == wcis_id) and
                    (not company or doc["company"].lower() == company.lower())), None)

        if not doc:
            return {"response": "Sorry, no record found."}

        # Now process the document with the corresponding intent handler
        intent = None
        vec = model.encode([user_text])
        sims = cosine_similarity(vec, intent_embeddings)[0]
        best_match_idx = int(np.argmax(sims))
        intent = intent_labels[best_match_idx]
        confidence = sims[best_match_idx]
        logging.debug(f"Detected intent: {intent} (confidence: {confidence:.2f})")

        if intent in intent_handlers:
            if intent == "step_status":
                response = intent_handlers[intent](doc, user_text)
            elif intent == "who_is_customer":
                response = intent_handlers[intent](doc, wcis_id)
            else:
                response = intent_handlers[intent](doc)
        else:
            response = "Sorry, I couldn't understand your request."

        return {"response": response}

    except Exception as e:
        logging.error(f"Error during query handling: {str(e)}")
        return {"response": "An error occurred while processing your request. Please try again later."}
