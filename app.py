from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Initialize FastAPI
app = FastAPI()

# Load Model and Tokenizer
model_name = "./metallama2-7b-qa-tuned-merged"  # Update with your model's path
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model = model.to("cuda")  # Ensures the model uses the GPU

# Define Request and Response Schemas
class PredictionRequest(BaseModel):
    prompt: str

class PredictionResponse(BaseModel):
    response: str

# Define Prediction Endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        # Encode the prompt and generate response
        inputs = tokenizer(request.prompt, return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs, max_length=100)
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return PredictionResponse(response=response_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {e}")

# Run this script using Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)  # Binds to all interfaces for public access




