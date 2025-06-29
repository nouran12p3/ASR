from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-medium-hi32",  
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,  
    fp16=True,
    learning_rate=5e-4,
    warmup_steps=500,
    max_steps=4975,
    eval_strategy="steps",
    eval_steps=1250,
    save_strategy="steps",
    save_steps=2500,
    per_device_eval_batch_size=2,
    predict_with_generate=True,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=True,
)
