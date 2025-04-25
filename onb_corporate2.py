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

# Extract legalEntityName from message
def extract_legal_entity(text: str):
    try:
        companies = {doc["legalEntityName"] for doc in onboarding_data if "legalEntityName" in doc}
        for company in companies:
            if company.lower() in text.lower():
                return company
    except Exception as e:
        logging.error(f"Error extracting legalEntityName: {str(e)}")
    return None

# Extract WCIS ID from message
def extract_wcis_id(text: str):
    match = re.search(r"\b\d{6,}\b", text)
    return match.group(0) if match else None

# Intent-specific handlers
def handle_current_milestone(doc):
    entity = doc.get('legalEntityName', 'Unknown')
    milestone = doc.get('currentMilestone', 'Not Available')
    return f"Sure! Right now, {entity} is at the '{milestone}' milestone. Need more info on what's next?"

def handle_milestone_status(doc):
    entity = doc.get("legalEntityName", "Unknown")
    milestones = doc.get("milestones", [])
    if milestones:
        return f"Here’s how {entity} has been progressing: {', '.join(milestones)}. Want to know more about any specific milestone?"
    else:
        return f"I couldn't find any milestones listed for {entity}. Want me to double-check?"

def handle_accounts_milestone_status(doc):
    entity = doc.get("legalEntityName", "Unknown")
    milestones = doc.get("accountMilestones", [])
    if milestones:
        return f"As for account milestones, {entity} has the following updates: {', '.join(milestones)}. Shall I go deeper on any of these?"
    else:
        return f"Hmm, doesn't look like there are any account milestones available for {entity}."

def handle_internal_contacts(doc):
    entity = doc.get("legalEntityName", "Unknown")
    contacts = doc.get("internalContacts", [])
    if contacts:
        return f"Here’s who you can reach out to internally at {entity}: {', '.join(contacts)}. Want me to share their roles too?"
    else:
        return f"I couldn't spot any internal contacts for {entity}. Need me to poke around again?"

def handle_external_contacts(doc):
    entity = doc.get("legalEntityName", "Unknown")
    contacts = doc.get("externalContacts", [])
    if contacts:
        return f"External folks connected with {entity} include: {', '.join(contacts)}. Let me know if you need contact details."
    else:
        return f"Looks like there are no external contacts listed for {entity}. Want me to recheck?"

def handle_who_is_customer(doc, wcis_id):
    entity = doc.get("legalEntityName", "Unknown")
    return f"Got it! WCIS ID {wcis_id} is linked to {entity}. Anything else you'd like to know about them?"

# Casual small talk and greetings
small_talk_responses = {
    "hi": "Hey there! 👋 How can I help you today?",
    "hello": "Hello! I'm ObaasChat. What can I do for you today?",
    "hey": "Hey hey! 😊 Need help with onboarding info?",
    "how are you": "I'm doing great, thanks for asking! How about you?",
    "what can you do": "I can help you with onboarding status, milestones, contact info, and more. Just ask away!",
    "thanks": "You're very welcome! 😊 Anything else on your mind?",
    "thank you": "Glad to help! Let me know if there's more I can assist with.",
    "who are you": "I’m ObaasChat, your onboarding assistant! Here to make life a little easier. 😉",
    "what's up": "Not much, just chilling in the cloud ☁️. What can I do for you today?",
    "good morning": "Good morning! Hope your day’s off to a smooth start ☀️",
    "good evening": "Good evening! 🌆 How can I assist you before you wrap up your day?",
    "bye": "See you soon! Don’t hesitate to come back if you need anything. 👋",
    "good night": "Sweet dreams! 😴 Catch you later!"
}

# Intent dispatcher
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
        user_text = q.message.lower().strip()

        # Handle casual greetings and small talk
        for phrase, response in small_talk_responses.items():
            if phrase in user_text:
                return {"response": response}

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
            match_name = (legal_entity and d.get("legalEntityName", "").lower() == legal_entity.lower())
            if match_id or match_name:
                doc = d
                break

        if not doc:
            return {"response": "Hmm, I couldn't find any matching record. Maybe double-check the name or ID?"}

        if intent in intent_handlers:
            if intent == "who_is_customer":
                response = intent_handlers[intent](doc, wcis_id)
            else:
                response = intent_handlers[intent](doc)
        else:
            response = "I'm not quite sure what you're asking. Want to rephrase it a bit?"

        return {"response": response}

    except Exception as e:
        logging.error(f"Error during query handling: {str(e)}")
        return {"response": "Oops, something went wrong on my side. Mind trying again in a bit?"}
