import requests

payload = {
  "model": "tinyllama-1.1b-chat-v1.0",
  "messages": [
      {"role": "system", "content": "You are an assistant."},
      {"role": "user", "content": "How old are you?"}
  ],
  "functions": [
      {"name": "get_current_year", "description": "Return the current year", "parameters": {"type":"object","properties":{}}}
  ]
}

resp = requests.post("http://localhost:1234/v1/chat/completions", json=payload)
print(resp.json())
