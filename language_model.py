import sounddevice as sd
import numpy as np
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, BartForConditionalGeneration, BartTokenizer, T5ForConditionalGeneration, T5Tokenizer
import torch
import warnings
import time

# Suppress all warnings
warnings.filterwarnings("ignore")

# Load pre-trained Wav2Vec2 model
wav2vec_model = Wav2Vec2ForCTC.from_pretrained("./saved_models/finetune-v2")
wav2vec_processor = Wav2Vec2Processor.from_pretrained("./saved_models/finetune-v2")

# Load pre-trained BART model
bart_model = BartForConditionalGeneration.from_pretrained("./saved_models/bart")
bart_tokenizer = BartTokenizer.from_pretrained("./saved_models/bart")

# Load pre-trained T5 model
t5_model = T5ForConditionalGeneration.from_pretrained("./saved_models/t5")
t5_tokenizer = T5Tokenizer.from_pretrained("./saved_models/t5")

# Configuration for real-time audio
SAMPLE_RATE = 16000
CHUNK_DURATION = 5  # Duration of each audio chunk in seconds
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)

def transcribe_audio(audio_chunk, sampling_rate = SAMPLE_RATE):
    """
    Transcribe audio using Wav2Vec2.
    """
    # Start time
    start = time.time()

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

    # End time
    end = time.time()
    duration = int(end-start)
    print(f"[SPEECH MODEL] The execution time: {time.strftime('%H:%M:%S', time.gmtime(duration))}")


    return transcription[0]

# EMERGENCY_VERBS = ['call', 'help', "dispatch", "evacuate", "need", "arrest", "rescue", "bleed", "hurt", "steal", 
#                    "collapse", "scream", "hit", "occur"]
# EMERGENCY_NOUNS = ["emergency", "ambulance", "firefighter", "police", "accident", "siren", "evacuation", 
#                    "attack", "medic", "assistance", "fire department", "crash", "service", "danger", "fire", 
#                    "smoke", "threat", "alarm", "burglar", "suspect"]
# EMERGENCY_ADJECTIVES = ['urgent', 'critical', 'severe', 'dangerous', "suspicious"]
# EMERGENCY_ADVERBS = ['immediately', 'please', 'quickly', 'now', "stolen", "missing", "injured"]
# emergency_verbs_str = ", ".join(EMERGENCY_VERBS)
# emergency_nouns_str = ", ".join(EMERGENCY_NOUNS)
# emergency_adjectives_str = ", ".join(EMERGENCY_ADJECTIVES)
# emergency_adverbs_str = ", ".join(EMERGENCY_ADVERBS)

def process_text_with_bart(text):
    """
    Process text with BART (e.g., summarize, paraphrase, etc.).
    """
    # Start time
    start = time.time()


    prompt = f"fix: {text}"

    # Encode text for BART
    inputs = bart_tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)

    # Generate processed output
    summary_ids = bart_model.generate(inputs.input_ids, max_length=50, num_beams=5, repetition_penalty=2.5, early_stopping=True, temperature=0.7)
    processed_text = bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    # End time
    end = time.time()
    duration = int(end-start)
    print(f"[LANGUAGE MODEL] The execution time: {time.strftime('%H:%M:%S', time.gmtime(duration))}")

    return processed_text

def process_text_with_t5(text):
    """
    Process text with T5
    """
    # Start time
    start = time.time()


    prompt = f"Correct this sentence: {text}"

    inputs = t5_tokenizer.encode(prompt, return_tensors="pt", max_length=1024, truncation=True)
    corrected_ids = t5_model.generate(inputs, max_length=50, num_beams=5, repetition_penalty=2.5, early_stopping=True, temperature=0.7)
    corrected_sentence = t5_tokenizer.decode(corrected_ids[0], skip_special_tokens=True)

    # End time
    end = time.time()
    duration = int(end-start)
    print(f"[LANGUAGE MODEL] The execution time: {time.strftime('%H:%M:%S', time.gmtime(duration))}")

    corrected_sentences = corrected_sentence.split(":")[-1].split("!")[-1].split(".")[-1]
    corrected_sentence = corrected_sentences.strip()

    return corrected_sentence.upper()

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
    print("Processed Text with BART:", processed_text)

    # Process transcription with t5
    processed_text = process_text_with_t5(transcription)
    print("Processed Text with t5:", processed_text)

if __name__ == "__main__":
    
    # REAL TIME PROCESS
    print("Starting real-time transcription and processing...")
    with sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, callback=callback, blocksize=CHUNK_SIZE
    ):
        print("Listening... Press Ctrl+C to stop.")
        while True:
            pass


  