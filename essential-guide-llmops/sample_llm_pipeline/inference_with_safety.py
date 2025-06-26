# PURPOSE: This script serves a fine-tuned open-source LLM (google/flan-t5-small)
#          using FastAPI for real-time Stack Overflow Q&A inference.
# TOOL: FastAPI, Hugging Face Transformers
# USE CASE: Operationalizing an LLM in production as described in Chapter 4 of the LLMOps Booklet

from fastapi import FastAPI, HTTPException
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Initialize FastAPI app
app = FastAPI()

# 🤖 Model Configuration
# --------------------------------------------------------------------------------------------------
# We use a fine-tuned version of google/flan-t5-small trained on Stack Overflow Python questions.
# The model is hosted on Hugging Face Hub under the name "hamedasgari20/stackoverflow-flan-t5-small".
MODEL_NAME = "hamedasgari20/stackoverflow-flan-t5-small"

# 🧾 Load Tokenizer and Model
# --------------------------------------------------------------------------------------------------
# Tokenizer converts text into numerical tokens understood by the model.
# Model is loaded from Hugging Face Hub; ensure internet access or model is cached locally.
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
except Exception as e:
    raise RuntimeError(f"Error loading model: {e}")

# 🚀 FastAPI Endpoint for Inference
# --------------------------------------------------------------------------------------------------
# POST endpoint `/generate` accepts:
# - `prompt`: input question (e.g., "How do I read a CSV file?")
# - `max_length`: optional parameter to control response length
@app.post("/generate")
def generate_response(prompt: str, max_length: int = 128):
    try:
        # 📥 Step 1: Tokenize input prompt
        inputs = tokenizer(prompt, return_tensors="pt")

        # 🧠 Step 2: Generate response using model
        outputs = model.generate(**inputs, max_length=max_length)

        # 📤 Step 3: Decode output tokens into human-readable text
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 📤 Return JSON-formatted response
        return {"response": response}
    except Exception as e:
        # 🔒 Handle any errors during inference
        raise HTTPException(status_code=500, detail=str(e))

# 🏃‍♂️ Run API Locally
# --------------------------------------------------------------------------------------------------
# This block allows you to run the API directly using Uvicorn.
# Use `python inference_with_fastapi.py` to start the server.
if __name__ == "__main__":
    import uvicorn

    # Start FastAPI server
    uvicorn.run(app, host="0.0.0.0", port=8000)
