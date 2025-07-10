from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer, EarlyStoppingCallback

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


trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=Train_Data,
    eval_dataset=Eval_Data,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=processor.feature_extractor,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

trainer.train()
