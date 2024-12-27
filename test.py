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

model = Wav2Vec2ForCTC.from_pretrained("./saved_models/fine-tune")
SAMPLE_RATE = 16000
processor = Wav2Vec2Processor.from_pretrained("./saved_models/fine-tune")

def transcribe(audio_waveform):
    """
    Transcribe a waveform array into text.
    """
    print("Transcribing audio waveform...")
    input_values = processor(audio_waveform, sampling_rate=16000, return_tensors="pt").input_values
    with torch.no_grad():
        logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    return processor.batch_decode(predicted_ids)[0]

def transcribe_wav_file(file_path):
    """
    Load a .wav file, resample if needed, and transcribe it.
    """
    # Load the .wav file
    waveform, sample_rate = torchaudio.load(file_path)

    # Resample if the audio sample rate is different from the model's
    if sample_rate != SAMPLE_RATE:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=SAMPLE_RATE)(waveform)
    
    # Convert to numpy array and flatten (if stereo, average channels)
    audio_data = waveform.mean(dim=0).numpy()
    
    # Transcribe
    transcription = transcribe(audio_data)
    return transcription

if __name__ == "__main__":
    file_path = "data/custom/3.wav"  # Replace with the path to your .wav file
    print("Transcribing the .wav file...")
    result = transcribe_wav_file(file_path)
    print("Transcription:", result)
    print(processor.tokenizer.get_vocab())