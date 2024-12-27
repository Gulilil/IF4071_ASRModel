import sounddevice as sd
import numpy as np
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

model = Wav2Vec2ForCTC.from_pretrained("./saved_models/final_model")
processor = Wav2Vec2Processor.from_pretrained("./saved_models/final_model")

SAMPLE_RATE = 16000
CHUNK_DURATION = 4
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)

def transcribe(audio_chunk):
    audio_chunk = (audio_chunk / np.max(np.abs(audio_chunk))).astype(np.float32)
    
    input_values = processor(audio_chunk, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_values

    with torch.no_grad():
        logits = model(input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    return transcription[0]

def callback(indata, frames, time, status):
    if status:
        print(f"Status: {status}")
    audio_chunk = indata[:, 0] 
    transcription = transcribe(audio_chunk)
    print("Transcription:", transcription)


if __name__ == "__main__":
    print("Starting real-time transcription...")
    with sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, callback=callback, blocksize=CHUNK_SIZE
    ):
        print("Listening... Press Ctrl+C to stop.")
        while True:
            pass
