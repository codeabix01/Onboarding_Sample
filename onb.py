import logging
import re
import json
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pymongo import MongoClient

# Logging setup
logging.basicConfig(level=logging.DEBUG)

# Initialize FastAPI app
app = FastAPI()

# Connect to MongoDB with error handling
try:
    client = MongoClient("mongodb://localhost:27017")
    logging.info("Connected to MongoDB successfully.")
except Exception as e:
    logging.error(f"Error connecting to MongoDB: {str(e)}")
    raise

# Load sentence-transformers model with error handling
try:
    model = SentenceTransformer('./local-model')
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model: {str(e)}")
    raise

# Load intent configuration (this could also come from a DB)
with open("intent_config.json", "r") as f:
    intent_config = json.load(f)

# Flatten examples and map back to intents
intent_examples = [(intent['intent'], ex) for intent in intent_config for ex in intent['examples']]
intent_phrases = [ex[1] for ex in intent_examples]
intent_labels = [ex[0] for ex in intent_examples]

# Precompute intent embeddings
intent_embeddings = model.encode(intent_phrases)

# Pydantic model for request
class UserQuery(BaseModel):
    message: str

# Extract company name from message
def extract_company(text: str, db):
    try:
        companies = set()
        for intent in intent_config:
            collection = db[intent['db']][intent['collection']]
            companies.update([doc["company"] for doc in collection.find({}, {"company": 1})])
        for company in companies:
            if company.lower() in text.lower():
                return company
    except Exception as e:
        logging.error(f"Error extracting company: {str(e)}")
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
        vec = model.encode([user_text])
        sims = cosine_similarity(vec, intent_embeddings)[0]
        best_match_idx = int(np.argmax(sims))
        intent = intent_labels[best_match_idx]
        confidence = sims[best_match_idx]
        logging.debug(f"Detected intent: {intent} (confidence: {confidence:.2f})")

        wcis_id_match = re.search(r"\b\d{6,}\b", user_text)
        wcis_id = wcis_id_match.group(0) if wcis_id_match else None
        company = extract_company(user_text, client)

        intent_meta = next((item for item in intent_config if item['intent'] == intent), None)
        if not intent_meta:
            return {"response": "Intent not configured."}

        collection = client[intent_meta['db']][intent_meta['collection']]
        query = {}
        if wcis_id:
            query["wcis_id"] = wcis_id
        if company:
            query["company"] = {"$regex": f"^{re.escape(company)}$", "$options": "i"}

        doc = collection.find_one(query)
        if not doc:
            return {"response": "Sorry, no record found."}

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
