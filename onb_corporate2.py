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

# Load sentence-transformers model
try:
    model = SentenceTransformer('./local-model')
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model: {str(e)}")
    raise

# Load intent configuration
with open("intent_config.json", "r") as f:
    intent_config = json.load(f)

# Load onboarding data from local JSON file
with open("onboarding_data.json", "r") as f:
    onboarding_data = json.load(f)

# Flatten examples and map back to intents
intent_examples = [(intent['intent'], ex) for intent in intent_config for ex in intent['examples']]
intent_phrases = [ex[1] for ex in intent_examples]
intent_labels = [ex[0] for ex in intent_examples]

# Precompute intent embeddings
intent_embeddings = model.encode(intent_phrases)

# Pydantic model for request
class UserQuery(BaseModel):
    message: str

# Extract LegalEntityName from message
def extract_legal_entity(text: str):
    try:
        companies = {doc["LegalEntityName"] for doc in onboarding_data if "LegalEntityName" in doc}
        for company in companies:
            if company.lower() in text.lower():
                return company
    except Exception as e:
        logging.error(f"Error extracting LegalEntityName: {str(e)}")
    return None

# Extract WCIS ID from message
def extract_wcis_id(text: str):
    match = re.search(r"\b\d{6,}\b", text)
    return match.group(0) if match else None

# Intent-specific handlers
def handle_current_milestone(doc):
    return f"Current milestone for {doc.get('LegalEntityName', 'Unknown')} is {doc.get('currentMilestone', 'Not Available')}."

def handle_milestone_status(doc):
    return {
        "LegalEntityName": doc.get("LegalEntityName", "Unknown"),
        "milestones": doc.get("milestones", [])
    }

def handle_accounts_milestone_status(doc):
    return {
        "LegalEntityName": doc.get("LegalEntityName", "Unknown"),
        "accountsMilestones": doc.get("accountsMilestones", [])
    }

def handle_internal_contacts(doc):
    return {
        "LegalEntityName": doc.get("LegalEntityName", "Unknown"),
        "internalContacts": doc.get("internalContacts", [])
    }

def handle_external_contacts(doc):
    return {
        "LegalEntityName": doc.get("LegalEntityName", "Unknown"),
        "externalContacts": doc.get("externalContacts", [])
    }

def handle_who_is_customer(doc, wcis_id):
    return f"WCIS ID {wcis_id} is associated with {doc.get('LegalEntityName', 'Unknown')}"

# Intent dispatcher with more specific logic
intent_handlers = {
    "current_milestone": handle_current_milestone,
    "milestone_status": handle_milestone_status,
    "accounts_milestone_status": handle_accounts_milestone_status,
    "internal_contacts": handle_internal_contacts,
    "external_contacts": handle_external_contacts,
    "who_is_customer": handle_who_is_customer
}

# Main API handler
@app.post("/query")
def handle_query(q: UserQuery):
    try:
        user_text = q.message
        vec = model.encode([user_text])
        sims = cosine_similarity(vec, intent_embeddings)[0]
        best_match_idx = int(np.argmax(sims))
        intent = intent_labels[best_match_idx]
        confidence = sims[best_match_idx]
        logging.debug(f"Detected intent: {intent} (confidence: {confidence:.2f})")

        wcis_id = extract_wcis_id(user_text)
        legal_entity = extract_legal_entity(user_text)

        # Search for matching document
        doc = None
        for d in onboarding_data:
            match_id = (wcis_id and d.get("wcisId") == wcis_id)
            match_name = (legal_entity and d.get("LegalEntityName", "").lower() == legal_entity.lower())
            if match_id or match_name:
                doc = d
                break

        if not doc:
            return {"response": "Sorry, no matching record found."}

        if intent in intent_handlers:
            # Additional check for accountMilestones to avoid misdirection
            if intent == "accounts_milestone_status":
                response = intent_handlers[intent](doc)
            elif intent == "internal_contacts":
                response = intent_handlers[intent](doc)
            elif intent == "external_contacts":
                response = intent_handlers[intent](doc)
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
