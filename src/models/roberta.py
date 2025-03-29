import torch
from torch.utils.data import DataLoader
from transformers import (
    RobertaTokenizer, 
    RobertaForSequenceClassification,
    AdamW, 
    get_scheduler,
    DataCollatorWithPadding
)
from datasets import Dataset
from tqdm import tqdm
import evaluate
import pandas as pd
import pickle

class RobertaTokenizer_imdb:
    """Class to handle data preparation for text classification"""
    
    def __init__(self, model_name="roberta-base"):
        """Initialize dataset with appropriate tokenizer"""
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer, return_tensors="pt")
    
    def load_data(self, file_path, max_length=25000):
        """Load data from a CSV file with a max length"""
        df = pd.read_csv(file_path)
        return df.head(max_length)
    
    def tokenize_function(self, examples):
        """Tokenization function for text samples"""
        return self.tokenizer(examples["comment"], padding="max_length", truncation=True)
    
    def prepare_dataset(self, df, text_column="comment", label_column="sentiment"):
        """Convert a pandas DataFrame into a processed dataset"""
        dataset = Dataset.from_pandas(df)
        
        tokenized_dataset = dataset.map(self.tokenize_function, batched=True)
        
        columns_to_remove = [col for col in tokenized_dataset.column_names 
                             if col not in [label_column] and col not in self.tokenizer.model_input_names]
        tokenized_dataset = tokenized_dataset.remove_columns(columns_to_remove)
        
        if label_column != "label":
            tokenized_dataset = tokenized_dataset.rename_column(label_column, "label")
        
        return tokenized_dataset
    
    def split_dataset(self, dataset, test_size=0.2, seed=42):
        """Divise un dataset en ensembles d'entraînement et de test"""
        split_datasets = dataset.train_test_split(test_size=test_size, seed=seed)
        return  split_datasets
    
    
    def create_dataloader(self, dataset, batch_size=8, shuffle=False):
        """Create a DataLoader for the dataset"""
        return DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            collate_fn=self.data_collator
        )

class RobertaModel_imdb:
    """Class to handle model training and evaluation"""
    
    def __init__(self, model_name="roberta-base", num_labels=2):
        """Initialize model"""
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available() 
            else "cpu"
        )
        print(f"Using device: {self.device}")
        
        # Load model
        self.model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

        self.model.to(self.device)
        
        # Evaluation metric
        self.metric = evaluate.load("accuracy")

    def load_model(self, filepath="roberta_model.pt"):
        """Load model with smaller .pt file"""
        self.model.load_state_dict(torch.load(filepath, map_location="cpu"))
        self.model.eval()  
    
    def train(self, train_dataloader, num_epochs=3, learning_rate=5e-5, gradient_accumulation_steps=4, model_filepath="roberta_imdb.pt", loss_filepath="roberta_training_losses.pkl"):
        """Train the model"""
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        
        # Learning rate scheduler
        num_training_steps = len(train_dataloader) * num_epochs
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps,
        )
        losses = []

        # Training loop
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            self.model.train()
            loop = tqdm(train_dataloader, leave=True)
            optimizer.zero_grad()
            epoch_losses = []  # Pour collecter les pertes par époque
            
            for i, batch in enumerate(loop):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss / gradient_accumulation_steps
                loss.backward()
                
                # Stocker la valeur de loss actuelle (en multipliant par gradient_accumulation_steps pour obtenir la valeur réelle)
                current_loss = loss.item() * gradient_accumulation_steps
                epoch_losses.append(current_loss)
                
                if (i + 1) % gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    lr_scheduler.step()
                
                loop.set_postfix(loss=current_loss)
            
            # Ajouter les pertes de cette époque à la liste principale
            losses.append(epoch_losses)
            
            self.save(filepath=model_filepath)
        
        # Sauvegarder toutes les valeurs de loss dans un fichier pickle
        with open(loss_filepath, 'wb') as f:
            pickle.dump(losses, f)
            
            print("Training complete")
    
    def evaluate(self, dataloader, split_name="Validation"):
        """Evaluate the model on a dataset"""
        self.model.eval()  # Set model to evaluation mode

        # Initialize storage for predictions and references
        predictions_list = []
        references_list = []

        for batch in tqdm(dataloader, desc=f"Evaluating on {split_name}"):
            batch = {k: v.to(self.device) for k, v in batch.items()}

            with torch.no_grad():
                outputs = self.model(**batch)
                predictions = torch.argmax(outputs.logits, dim=-1)

                # Store predictions and actual labels
                predictions_list.extend(predictions.cpu().numpy())
                references_list.extend(batch["labels"].cpu().numpy())

        # Compute accuracy
        accuracy = self.metric.compute(predictions=predictions_list, references=references_list)

        print(f"{split_name} Accuracy: {accuracy['accuracy']:.4f}")
        return accuracy

    
    def save(self, filepath="roberta_imdb.pt"):
        """Save only the model weights to reduce size"""
        torch.save(self.model.state_dict(), filepath)
        print(f"Model weights saved to {filepath}")