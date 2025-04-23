import requests

url = "http://127.0.0.1:8000/query"
data = {"message": "How far is Apple with WCIS ID 123456?"}
response = requests.post(url, json=data)
print(response.json())
