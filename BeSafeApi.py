from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pathlib import Path
from typing import List
import numpy as np
import onnxruntime as ort
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time
import logging
from transformers import AutoTokenizer
from contextlib import asynccontextmanager

# ---------------------------------------------------------------------------
#  BASIC CONFIGURATION
# ---------------------------------------------------------------------------


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("threat_api")

# Global variables
onnx_session = None
tokenizer = None
_executor = ThreadPoolExecutor(max_workers=4)

MODEL_PATH = str(Path("threatnet_Production.onnx"))

TOKENIZER_NAME = "bert-base-uncased"   # 🔁 Change this to your actual tokenizer


# ---------------------------------------------------------------------------
#  PYDANTIC MODELS (input/output schemas)
# ---------------------------------------------------------------------------
class TextInput(BaseModel):
    text: str = Field(..., description="Input text from speech-to-text", min_length=1)

class PredictionResponse(BaseModel):
    text: str
    is_threat: bool
    confidence: float
    label: str
    processing_time_ms: float

class ModelInfo(BaseModel):
    model_loaded: bool
    input_name: str
    output_name: str
    model_path: str


# ---------------------------------------------------------------------------
#  TEXT PREPROCESSING
# ---------------------------------------------------------------------------
def preprocess_text(text: str) -> str:
    """Normalize text: lowercase and strip whitespace."""
    import re
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
#  TOKENIZATION (using Hugging Face tokenizer)
# ---------------------------------------------------------------------------
def tokenize_texts(texts: List[str], max_length: int = 128):
    """Tokenize a list of texts into model-ready numpy arrays."""
    encodings = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="np"
    )
    return encodings


# ---------------------------------------------------------------------------
#  LOAD MODEL AT STARTUP
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global onnx_session, tokenizer

    logger.info("Starting up Threat Detection API...")
    try:
        onnx_session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load ONNX model: {e}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
        logger.info("Tokenizer loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")

    yield  # <-- Application runs between startup and shutdown

    logger.info("Cleaning up resources...")
    onnx_session = None
    tokenizer = None
    logger.info("Shutdown complete.")


app = FastAPI(
    title="Threat Detection API",
    description="API for detecting threatening content in text using an ONNX model",
    version="2.0.0",
    lifespan=lifespan
)

# ---------------------------------------------------------------------------
#  ASYNC INFERENCE WRAPPER
# ---------------------------------------------------------------------------
async def run_inference(inputs: dict):
    """Run ONNX inference asynchronously in a thread pool."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        _executor, lambda: onnx_session.run(None, inputs)
    )


# ---------------------------------------------------------------------------
#  ROOT HEALTH CHECK
# ---------------------------------------------------------------------------
@app.get("/", tags=["Health"])
async def root():
    """Simple health check endpoint."""
    return {
        "status": "online",
        "model_loaded": onnx_session is not None,
        "tokenizer_loaded": tokenizer is not None
    }


# ---------------------------------------------------------------------------
#  MODEL INFO
# ---------------------------------------------------------------------------
@app.get("/model/info", response_model=ModelInfo, tags=["Model"])
async def get_model_info():
    if onnx_session is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    inputs = onnx_session.get_inputs()
    outputs = onnx_session.get_outputs()

    return {
        "model_loaded": True,
        "input_name": inputs[0].name if inputs else "N/A",
        "output_name": outputs[0].name if outputs else "N/A",
        "model_path": MODEL_PATH,
    }


# ---------------------------------------------------------------------------
#  SINGLE PREDICTION
# ---------------------------------------------------------------------------
@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_threat(input_data: TextInput):
    if onnx_session is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model or tokenizer not loaded.")

    start_time = time.time()

    try:
        cleaned_text = preprocess_text(input_data.text)
        enc = tokenize_texts([cleaned_text])

        input_name = onnx_session.get_inputs()[0].name
        ort_inputs = {input_name: enc["input_ids"].astype(np.int64)}

        outputs = await run_inference(ort_inputs)
        logits = outputs[0][0]  # first batch

        # Process output (binary classification)
        if len(logits) == 2:
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / exp_logits.sum()
            threat_prob = float(probs[1])
        else:
            threat_prob = float(1 / (1 + np.exp(-logits)))
        is_threat = threat_prob > 0.5

        duration_ms = (time.time() - start_time) * 1000

        return PredictionResponse(
            text=input_data.text,
            is_threat=is_threat,
            confidence=round(threat_prob if is_threat else 1 - threat_prob, 4),
            label="THREAT" if is_threat else "NON-THREAT",
            processing_time_ms=round(duration_ms, 2)
        )

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# ---------------------------------------------------------------------------
#  BATCH PREDICTION (efficient batching)
# ---------------------------------------------------------------------------
@app.post("/predict/batch", tags=["Prediction"])
async def predict_batch(texts: List[str]):
    """Efficiently predict threats for multiple texts."""
    if onnx_session is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model or tokenizer not loaded.")

    start_time = time.time()

    cleaned_texts = [preprocess_text(t) for t in texts]
    enc = tokenize_texts(cleaned_texts)
    input_name = onnx_session.get_inputs()[0].name
    ort_inputs = {input_name: enc["input_ids"].astype(np.int64)}

    try:
        outputs = await run_inference(ort_inputs)
        logits = outputs[0]  # shape: (batch, num_classes or 1)
        results = []

        for i, text in enumerate(texts):
            logit = logits[i]
            if len(logit) == 2:
                exp_logits = np.exp(logit - np.max(logit))
                probs = exp_logits / exp_logits.sum()
                threat_prob = float(probs[1])
            else:
                threat_prob = float(1 / (1 + np.exp(-logit)))
            is_threat = threat_prob > 0.5

            results.append({
                "text": text,
                "is_threat": is_threat,
                "confidence": round(threat_prob if is_threat else 1 - threat_prob, 4),
                "label": "THREAT" if is_threat else "NON-THREAT"
            })

        return {
            "total": len(results),
            "processing_time_ms": round((time.time() - start_time) * 1000, 2),
            "predictions": results
        }

    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


# ---------------------------------------------------------------------------
#  MAIN (local dev only)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    import threading
    import time
    
    def run_server():
        uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
        
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    time.sleep(5)  # Keep the main thread alive for a short time
    
