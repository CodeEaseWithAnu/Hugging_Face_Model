from fastapi import FastAPI
from transformers import pipeline
import uvicorn

model_name = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
model_folder = "model/" + model_name.replace("/", "_")

ai = pipeline("text-generation", model=model_folder)

app = FastAPI()


@app.get("/ask")
def ask(question: str):
    message = [{"role": "user", "content": question}]
    result = ai(message, max_new_tokens=100)
    return {"answer": result[0]["generated_text"][-1]["content"]}


uvicorn.run(app, host="127.0.0.1", port=8000)
