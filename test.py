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
import pandas as pd
from thefuzz import fuzz
from thefuzz import process
import sys

model = Wav2Vec2ForCTC.from_pretrained("./saved_models/wav2vec2")
SAMPLE_RATE = 16000
processor = Wav2Vec2Processor.from_pretrained("./saved_models/wav2vec2")


def calculate_edit_operations(str1, str2):
    """
    Calculate the number of insertions, deletions, and substitutions needed to transform str1 into str2.
    """
    m, n = len(str1), len(str2)
    dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]

    # Initialize DP table
    for i in range(m + 1):
        dp[i][0] = i  # Cost of all deletions
    for j in range(n + 1):
        dp[0][j] = j  # Cost of all insertions

    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]  # No cost if characters match
            else:
                dp[i][j] = min(
                    dp[i - 1][j - 1],  # Substitution
                    dp[i - 1][j],      # Deletion
                    dp[i][j - 1]       # Insertion
                ) + 1

    # Trace back to count operations
    i, j = m, n
    insertions = deletions = substitutions = 0

    while i > 0 and j > 0:
        if str1[i - 1] == str2[j - 1]:
            i -= 1
            j -= 1
        elif dp[i][j] == dp[i - 1][j - 1] + 1:
            substitutions += 1
            i -= 1
            j -= 1
        elif dp[i][j] == dp[i - 1][j] + 1:
            deletions += 1
            i -= 1
        elif dp[i][j] == dp[i][j - 1] + 1:
            insertions += 1
            j -= 1

    # Account for remaining characters
    insertions += j
    deletions += i

    return insertions, deletions, substitutions

def analyze(actual_text: str, detected_text: str) -> dict:
    actual_text_splitted = actual_text.replace(".", "").split(" ")
    detected_text_splitted = detected_text.split(" ")
    actual_text_length = len(actual_text_splitted)

    # Use the fuzz to get the ratio
    fuzz_ratio = fuzz.ratio(actual_text, detected_text)

    # Search for exact match:
    insertions, deletions, substitutions = calculate_edit_operations(actual_text_splitted, detected_text_splitted)
    error_rate = (insertions + substitutions + deletions) * 100 / (actual_text_length)

    result = {
        "actual_text" : actual_text,
        "detected_text" : detected_text,
        "insertions" : insertions, 
        "substitutions" : substitutions,
        "deletions" : deletions,
        "error_rate" : error_rate,
        "similarity_score" : fuzz_ratio
    }
    return result
            

def transcribe(audio_waveform):
    """
    Transcribe a waveform array into text.
    """
    # print("Transcribing audio waveform...")
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
    
    # HOW TO RUN:
    # Closed Experimental Condition
    # ```cmd
    # python test.py closed
    # ```
    #
    # Open Experimental Condition
    # ```cmd
    # python test.py open
    # ```

    if (sys.argv[1] == "closed"):
      # Closed Experimental Condition
      df = pd.read_csv("data_train.csv")
      result_data = []
      total_insertions = 0
      total_deletions = 0
      total_substitutions = 0
      total_error_rate = 0
      total_similarity_score = 0
      total_data = len(df)
      for idx, row in df.iterrows():
          file_path = row['audio']
          actual_text = row['text']

          # print("Transcribing the .wav file...")
          result = transcribe_wav_file(file_path)
          print(f"[PROCESSING] {file_path}")
          print("Transcription:", result)
          print("Actual Text:", actual_text.upper())
          # print(processor.tokenizer.get_vocab())
          analyze_result = analyze(actual_text.upper(), result)
          result_data.append(analyze_result)

          total_insertions += analyze_result['insertions']
          total_deletions += analyze_result['deletions']
          total_substitutions += analyze_result['substitutions']
          total_error_rate += analyze_result['error_rate']
          total_similarity_score += analyze_result['similarity_score']

      print("[OVERALL SCORE]")
      print(f"Average insertions: {total_insertions/total_data}")
      print(f"Average substitutions: {total_substitutions/total_data}")
      print(f"Average deletions: {total_deletions/total_data}")
      print(f"Average error rate: {total_error_rate/total_data}")
      print(f"Average similarity score: {total_similarity_score/total_data}")

      # result_df = pd.DataFrame(result_data)
      # result_df.to_csv("./result/result_data_test_closed.csv", index=False)
      # result_df.to_excel("./result/result_data_test_closed.xlsx", sheet_name="Result", index=False)

    elif (sys.argv[1] == "open"):
      # Open Experimental Condition
      df = pd.read_csv("data_test.csv")
      result_data = []
      total_insertions = 0
      total_deletions = 0
      total_substitutions = 0
      total_error_rate = 0
      total_similarity_score = 0
      total_data = len(df)
      for idx, row in df.iterrows():
          file_path = row['audio']
          actual_text = row['text']

          # print("Transcribing the .wav file...")
          result = transcribe_wav_file(file_path)
          print(f"[PROCESSING] {file_path}")
          print("Transcription:", result)
          print("Actual Text:", actual_text.upper())
          # print(processor.tokenizer.get_vocab())
          analyze_result = analyze(actual_text.upper(), result)
          result_data.append(analyze_result)

          total_insertions += analyze_result['insertions']
          total_deletions += analyze_result['deletions']
          total_substitutions += analyze_result['substitutions']
          total_error_rate += analyze_result['error_rate']
          total_similarity_score += analyze_result['similarity_score']

      print("[OVERALL SCORE]")
      print(f"Average insertions: {total_insertions/total_data}")
      print(f"Average substitutions: {total_substitutions/total_data}")
      print(f"Average deletions: {total_deletions/total_data}")
      print(f"Average error rate: {total_error_rate/total_data}")
      print(f"Average similarity score: {total_similarity_score/total_data}")

      result_df = pd.DataFrame(result_data)
      result_df.to_csv("./result/result_data_test_open.csv", index=False)
      result_df.to_excel("./result/result_data_test_open.xlsx", sheet_name="Result", index=False)
        

