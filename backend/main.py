from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import librosa
import numpy as np
import io
import os

app = FastAPI()

# Enable CORS (change "*" to your frontend URL in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # e.g. "https://your-frontend.vercel.app"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Whisper processor and model
processor = WhisperProcessor.from_pretrained("Marwan-Kasem/whisper-medium-hi32")
model = WhisperForConditionalGeneration.from_pretrained("Marwan-Kasem/whisper-medium-hi32")

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Override the generation config
model.generation_config.forced_decoder_ids = None
model.generation_config.max_length = 225

@app.post("/upload")
async def upload_audio(audio: UploadFile = File(...)):
    try:
        # File size validation
        if audio.size and audio.size > 10 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="File is too large. Max size: 10MB.")

        print(f"Received file: {audio.filename} (size: {audio.size} bytes)")

        # Read and decode audio
        audio_data = await audio.read()
        if not audio_data:
            raise HTTPException(status_code=400, detail="Empty audio file received.")

        print("File read successfully.")

        # Load and resample to 16kHz
        audio_stream = io.BytesIO(audio_data)
        audio_array, sampling_rate = librosa.load(audio_stream, sr=16000)
        print(f"Audio loaded successfully. Sampling rate: {sampling_rate} Hz")

        # Normalize audio
        audio_array = librosa.util.normalize(audio_array)
        print("Audio normalized.")

        # Prepare input for Whisper
        inputs = processor(audio_array, sampling_rate=16000, return_tensors="pt")
        input_features = inputs.input_features.to(device)
        attention_mask = inputs.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        print("Audio preprocessed.")

        # Generate transcription
        with torch.no_grad():
            predicted_ids = model.generate(
                input_features,
                attention_mask=attention_mask,
                forced_decoder_ids=None
            )
            transcription = processor.decode(predicted_ids[0], skip_special_tokens=True)

        transcription = transcription.strip() if transcription else "Text not available"
        print(f"Transcription: {transcription}")

        return {"filename": audio.filename, "transcription": transcription}

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "Welcome to the ASR API"}

# For Railway: run Uvicorn if launched directly
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=port)
