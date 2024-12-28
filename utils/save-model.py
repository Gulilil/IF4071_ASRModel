from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, BartForConditionalGeneration, BartTokenizer
import torch

import os

print("START SAVING WAV2 MODEL")
model_name = "facebook/wav2vec2-base-960h"
model = Wav2Vec2ForCTC.from_pretrained(model_name)
processor = Wav2Vec2Processor.from_pretrained(model_name)

model.save_pretrained(os.path.join(os.getcwd(), "saved_models", "wav2vec2"))
processor.save_pretrained(os.path.join(os.getcwd(), "saved_models", "wav2vec2"))

print("START SAVING BART MODEL")

bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

bart_model.save_pretrained(os.path.join(os.getcwd(), "saved_models", "bart"))
bart_tokenizer.save_pretrained(os.path.join(os.getcwd(), "saved_models", "bart"))

print("Models and tokenizers have been saved successfully!")