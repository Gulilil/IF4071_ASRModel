import json
import torch
import torchaudio
from datasets import Dataset
from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    TrainingArguments,
    Trainer,
)

# Step 1: Load Dataset
def load_data(data_dir):
    with open(f"{data_dir}/transcripts.json", "r") as f:
        data = json.load(f)
    return Dataset.from_dict({
        "path": [f"{data_dir}/{item['path']}" for item in data],
        "transcription": [item["transcription"] for item in data]
    })

print("Loading dataset...")
train_dataset = load_data("data/train")

# Step 2: Preprocess Data
print("Preprocessing data...")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

def preprocess(batch):
    audio = torchaudio.load(batch["path"])[0]
    if len(audio.shape) > 1:  # Stereo to mono
        audio = torch.mean(audio, dim=0)
    if audio.shape[0] != 16000:  # Resample to 16kHz
        audio = torchaudio.transforms.Resample(orig_freq=audio.shape[0], new_freq=16000)(audio)
    batch["input_values"] = processor(audio.squeeze().numpy(), sampling_rate=16000, return_tensors="pt").input_values[0]
    batch["labels"] = processor.tokenizer(batch["transcription"]).input_ids
    return batch

train_dataset = train_dataset.map(preprocess, remove_columns=["path", "transcription"])

# Step 3: Load Pre-Trained Model
print("Loading Wav2Vec2 model...")
model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-base-960h",
    vocab_size=processor.tokenizer.vocab_size
)

# Step 4: Training Configuration
print("Setting up training arguments...")
training_args = TrainingArguments(
    output_dir="./wav2vec2-finetuned",
    evaluation_strategy="no",  # No evaluation during training
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,
    num_train_epochs=5,
    save_steps=500,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=10,
)

# Step 5: Fine-Tuning with Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=processor.feature_extractor,
)

print("Starting fine-tuning...")
trainer.train()

# Step 6: Save Fine-Tuned Model
print("Saving fine-tuned model...")
model.save_pretrained("./wav2vec2-finetuned")
processor.save_pretrained("./wav2vec2-finetuned")

# Step 7: Evaluation Function
def transcribe(audio_path):
    print(f"Transcribing: {audio_path}")
    waveform, _ = torchaudio.load(audio_path)
    input_values = processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt").input_values
    with torch.no_grad():
        logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    return processor.batch_decode(predicted_ids)[0]

# Example Usage
audio_file = "data/train/audio1.wav"
print("Example Transcription:")
print(transcribe(audio_file))