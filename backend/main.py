from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import librosa
import numpy as np
import io

app = FastAPI()

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Whisper processor and model (medium version from Hugging Face)
processor = WhisperProcessor.from_pretrained("Marwan-Kasem/whisper-medium-hi32")
model = WhisperForConditionalGeneration.from_pretrained("Marwan-Kasem/whisper-medium-hi32")

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Override the generation config
model.generation_config.forced_decoder_ids = None  # Remove forced language decoding
model.generation_config.max_length = 225  # Optional: Cap output length

@app.post("/upload")
async def upload_audio(audio: UploadFile = File(...)):
    try:
        # File size validation (max ~10MB)
        if audio.size and audio.size > 10 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="File is too large. Max size: 10MB.")

        print(f"Received file: {audio.filename} (size: {audio.size} bytes)")

        # Read audio data
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

        # Prepare for model
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
                forced_decoder_ids=None  # Disable language forcing
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


# to run the backend use:
# cd C:\Users\Smart\Desktop\asr_project_website\backend
# uvicorn main:app --reload