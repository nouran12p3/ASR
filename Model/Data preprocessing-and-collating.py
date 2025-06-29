import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union

def prepare_dataset(batch):
    audio = batch["audio"]
    # Extract log‑Mel features as a torch.Tensor
    inputs = feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
        return_tensors="pt"
    )
    # inputs.input_features is shape (1, seq_len, feature_dim)
    batch["input_features"] = inputs.input_features.squeeze(0)

    # Tokenize your transcripts to a torch.LongTensor of token IDs
    labels = processor.tokenizer(
        batch["raw_text"],
        return_tensors="pt"
    ).input_ids
    batch["labels"] = labels.squeeze(0)

    return batch
def filter_long_samples(batch):
    labels = processor.tokenizer(
        batch["raw_text"],
        return_tensors="pt"
    ).input_ids
    return labels.shape[1] <= 448

Train_Data = Train_Data.filter(filter_long_samples, num_proc=4)

eval_Data = eval_Data.filter(filter_long_samples, num_proc=4)

Train_Data = Train_Data.remove_columns(["audio_id", "gender", "speaker_id", "duration","split","accent"])

eval_Data = eval_Data.remove_columns(["audio_id", "gender", "speaker_id", "duration","split","accent"])

Train_Data = Train_Data.map(
    prepare_dataset,
    remove_columns=["audio", "raw_text"],)

Eval_Data = vali_data.map(
    prepare_dataset,
    remove_columns=["audio", "raw_text"],)


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any  # WhisperProcessor
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
    
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
    
        labels = labels_batch["input_ids"].masked_fill(labels_batch["attention_mask"].ne(1), -100)
    
        if (labels[:, 0] == self.decoder_start_token_id).all():
            labels = labels[:, 1:]
    
        batch["labels"] = labels
    
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)
