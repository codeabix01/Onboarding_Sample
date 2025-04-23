from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017")
db = client["onboarding"]
collection = db["clients"]

collection.insert_one({
    "company": "Apple",
    "wcis_id": "123456",
    "steps": {
        "KYC": "pending",
        "AccountOpening": "complete",
        "LegalEntity": "complete"
    },
    "status": "70%"
})

print("Sample data inserted successfully.")
