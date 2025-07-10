from transformers import (WhisperFeatureExtractor,AutoTokenizer,WhisperProcessor,WhisperForConditionalGeneration)

feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-medium")
tokenizer = AutoTokenizer.from_pretrained("openai/whisper-medium", language="English", task="transcribe")
processor = WhisperProcessor.from_pretrained("openai/whisper-medium", language="English", task="transcribe")

model_name = "openai/whisper-medium"  # Replace with your model checkpoint
model = WhisperForConditionalGeneration.from_pretrained(model_name)
