from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, BartForConditionalGeneration, BartTokenizer, T5ForConditionalGeneration, T5Tokenizer
import torch

import os

print("START SAVING WAV2 MODEL")
model_name = "facebook/wav2vec2-base-960h"
model = Wav2Vec2ForCTC.from_pretrained(model_name)
processor = Wav2Vec2Processor.from_pretrained(model_name)

model.save_pretrained(os.path.join(os.getcwd(), "saved_models", "wav2vec2"))
processor.save_pretrained(os.path.join(os.getcwd(), "saved_models", "wav2vec2"))

print("START SAVING BART MODEL")
bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")
bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")

bart_model.save_pretrained(os.path.join(os.getcwd(), "saved_models", "bart"))
bart_tokenizer.save_pretrained(os.path.join(os.getcwd(), "saved_models", "bart"))

print("START SAVING T5 LARGE SPELL")
t5_model = T5ForConditionalGeneration.from_pretrained("t5-large")
t5_tokenizer = T5Tokenizer.from_pretrained("t5-large")

t5_model.save_pretrained(os.path.join(os.getcwd(), "saved_models", "t5"))
t5_tokenizer.save_pretrained(os.path.join(os.getcwd(), "saved_models", "t5"))

print("Models and tokenizers have been saved successfully!")