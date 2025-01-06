import torch
import numpy as np
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from datasets import load_dataset, Audio, DatasetDict
from dataclasses import dataclass
from typing import Dict, List, Union
import torch.nn.functional as F
from transformers import TrainingArguments, Trainer
from jiwer import wer

class Wav2Vec2Trainer:
    def __init__(self, model_path: str, output_dir: str):
        self.model_path = model_path
        self.output_dir = output_dir
        
        # Check if MPS is available
        # self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        if (torch.cuda.is_available()):
              print(f"CUDA device count: {torch.cuda.device_count()}")
              print(f"Current device: {torch.cuda.current_device()}")
              print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        
        self.model = Wav2Vec2ForCTC.from_pretrained(model_path)
        self.processor = Wav2Vec2Processor.from_pretrained(model_path)
        
        # Move model to MPS if available
        self.model = self.model.to(self.device)

    def prepare_dataset(self, batch):
        audio = batch["audio"]
        
        # Process audio
        input_values = self.processor(
            audio["array"], 
            sampling_rate=audio["sampling_rate"],
            padding=True,
            return_tensors="pt"
        ).input_values
        
        batch["input_values"] = input_values.squeeze().numpy()
        
        # Process text without using as_target_processor
        batch["labels"] = self.processor.tokenizer(
            batch["sentence"].lower(), 
            padding=True, 
            return_tensors="pt"
        ).input_ids.squeeze().numpy()
        
        return batch

    def compute_metrics(self, pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        pred.label_ids[pred.label_ids == -100] = self.processor.tokenizer.pad_token_id

        pred_str = self.processor.batch_decode(pred_ids)
        label_str = self.processor.batch_decode(pred.label_ids, group_tokens=False)

        # Fixed the WER calculation
        wer_metric = wer(truth=label_str, hypothesis=pred_str)
        return {"wer": wer_metric}

    def get_data_collator(self):
        return DataCollatorCTCWithPadding(processor=self.processor)

    def fine_tune(self):
        # Load dataset
        print("Loading dataset...")
        train_dataset = load_dataset("DTU54DL/common-native", split="train[:10%]")
        validation_dataset = load_dataset("DTU54DL/common-native", split="test[:10%]")

        
        print("Processing audio files...")
        dataset = train_dataset.cast_column("audio", Audio(sampling_rate=16000))

        print("Preparing dataset...")
        train_dataset = train_dataset.map(
            self.prepare_dataset,
            remove_columns=train_dataset.column_names,
            batch_size=1,
            num_proc=1,
            desc="Processing training data"
        )

        print("Processing validation dataset...")
        validation_dataset = validation_dataset.map(
            self.prepare_dataset,
            remove_columns=validation_dataset.column_names,
            batch_size=1,
            num_proc=1,
            desc="Processing validation data"
        )

        data_collator = self.get_data_collator()

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=3e-5,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            per_device_eval_batch_size=2,
            num_train_epochs=10,
            warmup_ratio=0.1,
            logging_steps=10,
            load_best_model_at_end=True,
            metric_for_best_model="wer",  # Changed from "accuracy" to "wer"
            greater_is_better=False,  # Add this because lower WER is better
        )
        
        print("INI DATASET", dataset.column_names)
        print("Initializing trainer...")
        trainer = Trainer(
            model=self.model,
            data_collator=data_collator,
            args=training_args,
            compute_metrics=self.compute_metrics,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            tokenizer=self.processor.feature_extractor,
        )

        print("Starting training...")
        trainer.train()

        print("Saving model...")
        self.model.save_pretrained(f"{self.output_dir}/final_model")
        self.processor.save_pretrained(f"{self.output_dir}/final_model")
        print("Training completed!")


@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Get max length of input_values
        max_length = max(len(feature["input_values"]) for feature in features)
        
        # Pad input_values
        batch_input_values = []
        for feature in features:
            input_value = feature["input_values"]
            padding_length = max_length - len(input_value)
            if padding_length > 0:
                input_value = np.pad(input_value, (0, padding_length), 'constant', constant_values=0.0)
            batch_input_values.append(input_value)
            
        batch = {
            "input_values": torch.tensor(batch_input_values, dtype=torch.float32),
            "attention_mask": torch.ones(len(batch_input_values), max_length)
        }

        # Pad labels
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(label_feature["input_ids"]) for label_feature in label_features],
            batch_first=True,
            padding_value=self.processor.tokenizer.pad_token_id
        )

        batch["labels"] = labels_batch
        return batch


if __name__ == "__main__":
    model_path = "./saved_models/wav2vec2"
    output_dir = "./fine_tuned_comvoi"
    
    trainer = Wav2Vec2Trainer(model_path, output_dir)
    trainer.fine_tune()