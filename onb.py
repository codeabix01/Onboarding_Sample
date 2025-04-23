import logging
import re
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
    db = client["onboarding"]
    collection = db["clients"]
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

# Pydantic model for request
class UserQuery(BaseModel):
    message: str

# Training data for intent classification
intent_examples = [
    ("onboarding_status", "How far is Apple with WCIS ID 123456?"),
    ("onboarding_status", "How much percentage Adobe is onboarded?"),
    ("pending_steps", "What is pending for Apple with WCIS ID 123456?"),
    ("step_status", "Is KYC completed for Apple with WCIS ID 123456?"),
    ("who_is_customer", "Who is WCIS ID 123456 registered to?")
]

intent_phrases = [ex[1] for ex in intent_examples]
intent_labels = [ex[0] for ex in intent_examples]

# Precompute intent embeddings
intent_embeddings = model.encode(intent_phrases)

# Extract company name from message
def extract_company(text: str):
    try:
        known_companies = [doc["company"] for doc in collection.find({}, {"company": 1})]
        for company in known_companies:
            if company.lower() in text.lower():
                return company
    except Exception as e:
        logging.error(f"Error extracting company: {str(e)}")
    return None

# FastAPI endpoint for handling queries
@app.post("/query")
def handle_query(q: UserQuery):
    try:
        logging.debug(f"Received message: {q.message}")

        user_text = q.message

        # Encode user query
        vec = model.encode([user_text])
        sims = cosine_similarity(vec, intent_embeddings)[0]
        best_match_idx = int(np.argmax(sims))
        intent = intent_labels[best_match_idx]
        confidence = sims[best_match_idx]
        logging.debug(f"Detected intent: {intent} (confidence: {confidence:.2f})")

        # Extract WCIS ID and company
        wcis_id_match = re.search(r"\b\d{6,}\b", user_text)
        wcis_id = wcis_id_match.group(0) if wcis_id_match else None
        company = extract_company(user_text)
        logging.debug(f"Extracted WCIS ID: {wcis_id}, Company: {company}")

        # MongoDB query
        query = {}
        if wcis_id:
            query["wcis_id"] = wcis_id
        if company:
            query["company"] = {"$regex": f"^{re.escape(company)}$", "$options": "i"}

        logging.debug(f"MongoDB Query: {query}")
        doc = collection.find_one(query)

        # Handle no match
        if not doc:
            logging.info("No record found in the database.")
            return {"response": "Sorry, no record found."}

        # Build response based on intent
        if intent == "onboarding_status":
            response = f"{doc.get('company', 'Unknown')} onboarding is {doc.get('status', 'unknown')} complete."
        elif intent == "pending_steps":
            pending = [step for step, status in doc.get("steps", {}).items() if status != "complete"]
            response = f"Pending steps for {doc.get('company', 'Unknown')}: {', '.join(pending) if pending else 'None'}"
        elif intent == "step_status":
            for step in ["KYC", "AccountOpening", "LegalEntity"]:
                if step.lower() in user_text.lower():
                    response = f"{step} status for {doc.get('company', 'Unknown')}: {doc.get('steps', {}).get(step, 'unknown')}"
                    break
            else:
                response = "Step not recognized."
        elif intent == "who_is_customer":
            response = f"WCIS ID {wcis_id} belongs to {doc.get('company', 'Unknown')}."
        else:
            response = "Sorry, I couldn't understand your request."

        logging.info(f"Response: {response}")
        return {"response": response}

    except Exception as e:
        logging.error(f"Error during query handling: {str(e)}")
        return {"response": "An error occurred while processing your request. Please try again later."}
