import requests
files = {"file": open("sample-zh.wav","rb")}
data = {"model":"Qwen3-ASR-0.6B-Q8_0.gguf","language":"zh"}
resp = requests.post("http://localhost:8080/v1/audio/transcriptions", files=files, data=data, headers={"Authorization":"Bearer sk-no-key-required"})
print(resp.status_code); print(resp.headers); print(resp.text)