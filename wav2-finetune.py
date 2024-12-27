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
    def __init__(self, model_path: str, dataset_path: str, output_dir: str):
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        
        # Check if MPS is available
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
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
            batch["text"].lower(), 
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
        raw_dataset = load_dataset("csv", data_files=self.dataset_path)
        
        # Create a validation split manually
        train_test = raw_dataset["train"].train_test_split(
            test_size=0.5,  # Split 50-50 since we have a small dataset
            shuffle=True,
            seed=42
        )
        
        # Create a new dataset dictionary with both splits
        dataset = DatasetDict({
            'train': train_test['train'],
            'validation': train_test['test']
        })
        
        print(f"Train size: {len(dataset['train'])}")
        print(f"Validation size: {len(dataset['validation'])}")
        
        print("Processing audio files...")
        dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

        print("Preparing dataset...")
        dataset = dataset.map(
            self.prepare_dataset,
            remove_columns=dataset.column_names["train"],
            batch_size=1,
            num_proc=1,
            desc="Processing audio files"
        )

        data_collator = self.get_data_collator()

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            group_by_length=True,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=4,
            evaluation_strategy="steps",
            save_strategy="steps",
            num_train_epochs=50,
            save_steps=10,
            eval_steps=10,
            logging_steps=5,
            learning_rate=1e-4,
            weight_decay=0.005,
            warmup_steps=20,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="wer",
            greater_is_better=False,
            logging_dir=f"{self.output_dir}/logs",
            no_cuda=True
        )

        print("Initializing trainer...")
        trainer = Trainer(
            model=self.model,
            data_collator=data_collator,
            args=training_args,
            compute_metrics=self.compute_metrics,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
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
    dataset_path = "data.csv"
    output_dir = "./fine_tuned_wav2vec2"
    
    trainer = Wav2Vec2Trainer(model_path, dataset_path, output_dir)
    trainer.fine_tune()