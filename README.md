---
license: mit
datasets:
- westbrook/English_Accent_DataSet
language:
- en
metrics:
- wer
base_model:
- openai/whisper-medium
pipeline_tag: automatic-speech-recognition
library_name: adapter-transformers
---
#  README 
# 🗣️ Whisper-Medium-hi32: Fine-Tuned English ASR (Diverse Accents)

This repository hosts **`Whisper-Medium-hi32`**, a fine-tuned version of [OpenAI's Whisper-Medium](https://huggingface.co/openai/whisper-medium), developed by **Marwan Kasem** for **automatic speech recognition (ASR)** on clean, conversational English across diverse global accents.

---

## 📌 Overview

**Whisper-Medium-hi32** is optimized for:

- 📞 **Conversational English**
- 🌍 **Diverse Accents** (UK, Irish, American, etc.)
- 🧪 **Real-world scenarios** (non-studio recordings)

This model is trained to better handle the variability of everyday speech, making it ideal for **call center analytics**, **transcription services**, and **research in sociolinguistics or dialectal speech**.

---

## 📊 Evaluation Results

| Metric            | Value      |
|-------------------|------------|
| **Loss**          | 0.1971     |
| **WER (Word Error Rate)** | **16.52%** |

These results were obtained on a clean, diverse English evaluation set featuring a wide range of accents.

---

## 🧠 Model Details

- **Base Model:** `openai/whisper-medium`
- **Fine-Tuner:** Marwan Kasem
- **Frameworks:** PyTorch + HuggingFace Transformers + PEFT

---

## 🚀 Intended Use

This ASR model is ideal for:

- ✅ **Transcribing global English** in interviews, meetings, podcasts
- ✅ Enhancing **low-resource dialect ASR** pipelines
- ✅ Evaluating robustness to accentual variation

---

## ⚠️ Limitations

- ⚠️ Trained on a **subset of English** (not multilingual)
- ⚠️ May underperform on **noisy or overlapping speech**
- ⚠️ Dataset details are **currently under documentation**

---

## 🏋️ Training Configuration

| Hyperparameter               | Value        |
|------------------------------|--------------|
| `learning_rate`              | 5e-4         |
| `train_batch_size`           | 2            |
| `eval_batch_size`            | 2            |
| `gradient_accumulation_steps`| 4            |
| `total_train_batch_size`     | 8            |
| `optimizer`                  | AdamW        |
| `lr_scheduler`               | Linear       |
| `warmup_steps`               | 500          |
| `mixed_precision`            | Native AMP   |
| `seed`                       | 42           |

### 🧪 Training Results

| Epoch | Step | Train Loss | Val Loss | WER     |
|-------|------|------------|----------|---------|
| 0.20  | 1250 | 0.3184     | 0.2267   | 19.12%  |
| 0.40  | 2500 | 0.1957     | 0.2095   | 32.51%  |
| 0.60  | 3750 | 0.1969     | 0.1971   | **16.52%** ✅ |

---

## 🧰 Environment

- **PEFT**: `v0.14.0`
- **Transformers**: `v4.47.0`
- **PyTorch**: `v2.5.1+cu121`
- **Datasets**: `v3.3.1`
- **Tokenizers**: `v0.21.0`
---
## 💡 Usage Example (Python)

```python
# Libraries importing

import numpy as np
import librosa
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Load model and processor
processor = WhisperProcessor.from_pretrained("/kaggle/working/whisper_medium/Merged_Model")
model = WhisperForConditionalGeneration.from_pretrained("/kaggle/working/whisper_medium/Merged_Model")
model = model.to("cuda" if torch.cuda.is_available() else "cpu")

# Transcription function
def transcribe(stream, new_chunk):
    sr, y = new_chunk

    # Convert to mono if stereo
    if y.ndim > 1:
        y = y.mean(axis=1)

    y = y.astype(np.float32)
    y /= np.max(np.abs(y))

    if stream is not None:
        stream = np.concatenate([stream, y])
    else:
        stream = y

    # Convert to log-Mel spectrogram features
    inputs = processor.feature_extractor(
        stream,
        sampling_rate=sr,
        return_tensors="pt"
    )

    input_features = inputs.input_features.to(model.device)

    # Generate token ids
    predicted_ids = model.generate(
        input_features,
        return_timestamps=True,  # Required for >30s audio
    )

    # Decode to text
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    return stream, transcription

#  Load audio 
audio_path = "Your audio path"
y, sr = librosa.load(audio_path, sr=16000)

# Run transcription
stream, text = transcribe(None, (sr,&nbsp;y))
print(text) 
```
