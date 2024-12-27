import sounddevice as sd
import numpy as np
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, BartForConditionalGeneration, BartTokenizer
import torch
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

# Load pre-trained Wav2Vec2 model
wav2vec_model = Wav2Vec2ForCTC.from_pretrained("./saved_models/wav2vec2")
wav2vec_processor = Wav2Vec2Processor.from_pretrained("./saved_models/wav2vec2")

# Load pre-trained BART model
bart_model = BartForConditionalGeneration.from_pretrained("./saved_models/bart")
bart_tokenizer = BartTokenizer.from_pretrained("./saved_models/bart")

# Configuration for real-time audio
SAMPLE_RATE = 16000
CHUNK_DURATION = 5  # Duration of each audio chunk in seconds
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)

def transcribe_audio(audio_chunk, sampling_rate = SAMPLE_RATE):
    """
    Transcribe audio using Wav2Vec2.
    """
    # Normalize audio
    audio_chunk = (audio_chunk / np.max(np.abs(audio_chunk))).astype(np.float32)
    
    # Prepare input for Wav2Vec2 model
    input_values = wav2vec_processor(audio_chunk, sampling_rate=sampling_rate, return_tensors="pt").input_values
    
    # Perform inference
    with torch.no_grad():
        logits = wav2vec_model(input_values).logits

    # Decode predictions
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = wav2vec_processor.batch_decode(predicted_ids)
    return transcription[0]

def process_text_with_bart(text):
    """
    Process text with BART (e.g., summarize, paraphrase, etc.).
    """
    # Encode text for BART
    inputs = bart_tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)

    # Generate processed output
    summary_ids = bart_model.generate(inputs.input_ids, max_length=50, min_length=10, length_penalty=2.0, num_beams=4, early_stopping=True)
    processed_text = bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return processed_text

def callback(indata, frames, time, status):
    """
    Callback function for real-time audio processing.
    """
    if status:
        print(f"Status: {status}")
    
    audio_chunk = indata[:, 0]  # Use first channel
    transcription = transcribe_audio(audio_chunk)
    print("Raw Transcription:", transcription)

    # Process transcription with BART
    processed_text = process_text_with_bart(transcription)
    print("Processed Text:", processed_text)

if __name__ == "__main__":
    
    # REAL TIME PROCESS
    print("Starting real-time transcription and processing...")
    with sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, callback=callback, blocksize=CHUNK_SIZE
    ):
        print("Listening... Press Ctrl+C to stop.")
        while True:
            pass

    # # AUDIO PATH READ
    # Path to your audio file
    # audio_file_path = "data/test-1.wav"

    # # Load audio file
    # print(f"Loading audio file: {audio_file_path}")
    # waveform, sr = torchaudio.load(audio_file_path)

    # # Transcribe audio
    # print("Transcribing audio...")
    # transcription = transcribe_audio(waveform)
    # print("Raw Transcription:", transcription)

    # # Process transcription with BART
    # print("Processing transcription with BART...")
    # processed_text = process_text_with_bart(transcription)
    # print("Processed Text:", processed_text)