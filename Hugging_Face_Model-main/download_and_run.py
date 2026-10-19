from huggingface_hub import snapshot_download
from transformers import pipeline

model_name = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
model_folder = "model/" + model_name.replace("/", "_")

print("Downloading", model_name, "...")
snapshot_download(repo_id=model_name, local_dir=model_folder)
print("Model saved in", model_folder)

ai = pipeline("text-generation", model=model_folder)


while True:
    question = input("You: ")
    if question == "exit":
        break
    message = [{"role": "user", "content": question}]
    result = ai(message, max_new_tokens=100)
    print("AI:", result[0]["generated_text"][-1]["content"])
