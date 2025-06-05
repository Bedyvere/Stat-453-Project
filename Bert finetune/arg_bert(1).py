import json
import os
from typing import Dict, List, Optional, Tuple, Union  # Added for type hints

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, PreTrainedTokenizerBase
from tqdm import tqdm
import matplotlib.pyplot as plt
# --- Added: Import for F1 Score ---
from sklearn.metrics import precision_recall_fscore_support

# --- 1. Dataset Definition with modifications ---
class QADataset(Dataset):
    """
    Reads a JSONL file (or a JSON file for training) where each line contains:
    {
        "input": "...",  # Input text
        "label": 0 or 1  # Label (0 represents real, 1 represents false)
    }
    For training, if a JSON file is provided (e.g., rational_expert_en_train.json),
    it expects a structure like:
    {
       "en_train": [
           {
              "content": "...",
              "textual": "...",
              "label": 0 or 1
           },
           ...
       ]
    }
    and concatenates "content" and "textual" into one input text.
    """
    def __init__(self, file_path: str, tokenizer: PreTrainedTokenizerBase, max_length: int = 256):
        """
        Initializes the dataset.

        Args:
            file_path (str): Path to the data file. Either a JSONL file or a JSON file.
            tokenizer (PreTrainedTokenizerBase): Hugging Face Tokenizer for encoding text.
            max_length (int): Maximum sequence length after encoding.
        """
        super().__init__()
        self.samples: List[Dict] = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.jsonl':
            # Process file line by line (for validation and test files)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            data = json.loads(line.strip())
                            # Expecting each line to have "input" and "label"
                            if "content" in data and "label" in data:
                                self.samples.append(data)
                            else:
                                print(f"Warning: Skipping malformed line: {line.strip()}")
                        except json.JSONDecodeError:
                            print(f"Warning: Skipping line with JSON decode error: {line.strip()}")
            except FileNotFoundError:
                print(f"Error: File not found at {file_path}")
                raise
        elif ext == '.json':
            # Assume training file (e.g., rational_expert_en_train.json) with custom structure
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Expecting a key "en_train" that contains the samples
                    for item in data.get("ch_train", []):
                        if "content" in item and "textual" in item and "label" in item:
                            # Concatenate "content" and "textual" into a single text for encoding
                            combined_text = item["content"] + " " + item["textual"]
                            self.samples.append({
                                "content": combined_text,
                                "label": item["label"]
                            })
                        else:
                            print(f"Warning: Skipping item with missing keys: {item}")
            except FileNotFoundError:
                print(f"Error: File not found at {file_path}")
                raise
        else:
            raise ValueError(f"Unsupported file format: {ext}")

    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Gets the sample at the specified index and encodes it.

        Args:
            idx (int): Index of the sample.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing encoded data,
                                     with keys 'input_ids', 'attention_mask',
                                     'token_type_ids' (if applicable), 'label'.
        """
        data = self.samples[idx]
        text_a = data["content"]    # Use the concatenated input text
        label = data["label"]     # Label (0: real, 1: false)

        # Encode the text using the tokenizer
        encoding = self.tokenizer(
            text_a,
            padding='max_length',       # Pad to max_length
            truncation=True,            # Truncate sequences longer than max_length
            max_length=self.max_length,
            return_tensors='pt'         # Return PyTorch tensors
        )

        # Extract tensors from encoding, removing the batch dimension
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        token_type_ids = encoding.get("token_type_ids", None)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.squeeze(0)

        item = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": torch.tensor(label, dtype=torch.long)
        }
        if token_type_ids is not None:
            item["token_type_ids"] = token_type_ids

        return item

# --- 2. Custom Collate Function (Unchanged) ---
def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, Optional[torch.Tensor]]:
    """
    Collates a list of samples (from QADataset.__getitem__) into batch tensors.
    """
    input_ids = [item["input_ids"] for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]
    token_type_ids_list = [item.get("token_type_ids") for item in batch]
    labels = [item["label"] for item in batch]

    input_ids = torch.stack(input_ids, dim=0)
    attention_mask = torch.stack(attention_mask, dim=0)
    labels = torch.stack(labels, dim=0)

    if all(tid is not None for tid in token_type_ids_list):
        token_type_ids = torch.stack([tid for tid in token_type_ids_list if tid is not None], dim=0)
    else:
        token_type_ids = None

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        "labels": labels
    }

# --- 3. Model Definition (Unchanged) ---
class MyBertClassifier(nn.Module):
    def __init__(self, pretrained_model: str = "bert-base-uncased", dropout_prob: float = 0.1):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_model)
        hidden_size = self.bert.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, 384),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(384, 128),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(64, 2)
        )

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                token_type_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        return logits

# --- 4. Evaluation Function (Accuracy only) (REMOVED as functionality merged into test_model) ---
# This function is no longer strictly needed if test_model calculates all metrics directly.
# Keeping it might be useful for simpler accuracy checks elsewhere if needed in the future.
# For this request, we will calculate metrics directly in test_model.

# --- 4.1 Evaluation Function (Loss, Accuracy, and F1) ---
@torch.no_grad() # Disable gradient calculations
def evaluate_with_loss_and_f1(model: nn.Module,
                               data_loader: DataLoader,
                               device: torch.device,
                               criterion: nn.Module) -> Tuple[float, float, float, float]:
    """
    Evaluates the model's average loss, accuracy, and F1 scores (for class 1 and 0)
    on a given dataset.

    Args:
        model (nn.Module): The model to evaluate.
        data_loader (DataLoader): DataLoader containing the evaluation data.
        device (torch.device): The device (CPU or GPU) to run evaluation on.
        criterion (nn.Module): The loss function (e.g., nn.CrossEntropyLoss) to use.

    Returns:
        Tuple[float, float, float, float]: Average loss, accuracy, F1(Real=1), F1(False=0).
    """
    model.eval() # Set to evaluation mode
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    all_labels = []       # --- Added: Store all labels
    all_predictions = []  # --- Added: Store all predictions

    for batch in data_loader:
        # Move data to the designated device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        token_type_ids = batch.get('token_type_ids')
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)

        # Forward pass to get logits
        logits = model(input_ids, attention_mask, token_type_ids)
        # Calculate loss
        loss = criterion(logits, labels)
        # Accumulate total loss (multiply by batch size as some loss functions return mean)
        total_loss += loss.item() * labels.size(0)

        # Calculate accuracy
        predictions = torch.argmax(logits, dim=1)
        correct_predictions += (predictions == labels).sum().item()
        total_samples += labels.size(0)

        # --- Added: Collect labels and predictions for F1 calculation ---
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predictions.cpu().numpy())

    # Calculate average loss and accuracy
    average_loss = total_loss / total_samples if total_samples > 0 else 0.0
    accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0

    # --- Added: Calculate F1 scores using sklearn ---
    f1_real = 0.0
    f1_false = 0.0
    if total_samples > 0:
        # average=None gives per-class scores. labels=[0, 1] ensures output array has fixed size.
        # zero_division=0 handles cases where a class has no predictions or no true labels.
        precision, recall, fscore, support = precision_recall_fscore_support(
            all_labels,
            all_predictions,
            average=None,
            labels=[0, 1], # Ensure output for both classes 0 and 1
            zero_division=0
        )
        if len(fscore) >= 2: # Check if we got scores for both classes
             f1_false = fscore[0] # F1 for class 0 (False)
             f1_real = fscore[1]  # F1 for class 1 (Real)
        # Handle edge case if only one class score was returned despite labels=[0,1] (less likely)
        elif len(fscore) == 1:
            present_labels = set(all_labels) | set(all_predictions)
            if 0 in present_labels and 1 not in present_labels:
                 f1_false = fscore[0]
            elif 1 in present_labels and 0 not in present_labels:
                 f1_real = fscore[0]
            # If both 0 and 1 are somehow present but only one score returned, this logic might need review based on sklearn version/behavior

    model.train() # Restore training mode
    return average_loss, accuracy, f1_real, f1_false # --- Updated return values ---

# --- 5. Main Training Function (Updated for F1 reporting and plotting) ---
def train_model(train_file: str, valid_file: str,
                pretrained_model: str = 'bert-base-uncased',
                lr: float = 2e-5,
                epochs: int = 5,
                batch_size: int = 8,
                max_length: int = 512,
                output_dir: str = 'model_save',
                dropout_prob: float = 0.3,
                weight_decay: float = 0.01,
                early_stopping_patience: int = 3) -> float: # <--- Added: Patience for early stopping
    """
    Trains the BERT classification model, validates after each epoch (reporting F1 scores),
    saves the best model based on validation accuracy, implements early stopping,
    and plots training/validation curves (including F1 scores).

    Args:
        train_file (str): Path to the training data file (.jsonl).
        valid_file (str): Path to the validation data file (.jsonl).
        pretrained_model (str): Name or path of the pre-trained BERT model.
        lr (float): Learning rate.
        epochs (int): Maximum number of training epochs.
        batch_size (int): Batch size for training and validation.
        max_length (int): Maximum sequence length used by the tokenizer.
        output_dir (str): Directory to save the model, tokenizer, and plots.
        dropout_prob (float): Dropout probability in the model.
        weight_decay (float): Weight decay coefficient for the AdamW optimizer.
        early_stopping_patience (int): Number of epochs to wait for improvement before stopping.
                                       Set to a large value (e.g., epochs) or <= 0 to disable.

    Returns:
        float: The best validation accuracy achieved during training.
    """
    # Automatically select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Tokenizer
    tokenizer = BertTokenizer.from_pretrained(pretrained_model)

    # Create Datasets and DataLoaders
    print("Loading data...")
    train_dataset = QADataset(train_file, tokenizer, max_length=max_length)
    valid_dataset = QADataset(valid_file, tokenizer, max_length=max_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(valid_dataset)}")

    # Initialize Model
    print(f"Initializing model (based on {pretrained_model})...")
    # Ensure dropout_prob used here matches the intended value for the model
    model = MyBertClassifier(pretrained_model=pretrained_model, dropout_prob=dropout_prob)
    model.to(device)

    # Define Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # History tracking for metrics (Added F1 scores)
    history = {
        'train_loss': [], 'train_acc': [], 'train_f1_real': [], 'train_f1_false': [],
        'valid_loss': [], 'valid_acc': [], 'valid_f1_real': [], 'valid_f1_false': []
    }

    best_valid_acc = 0.0 # Track the best validation accuracy (used for saving model and early stopping)
    epochs_no_improve = 0 # <--- Added: Counter for epochs without validation accuracy improvement
    early_stop_triggered = False # <--- Added: Flag to indicate if early stopping was triggered

    # --- Start Training Loop ---
    print(f"Starting training for up to {epochs} epochs...")
    print(f"Early stopping enabled with patience={early_stopping_patience} (based on validation accuracy)")
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch + 1}/{epochs} ---")
        epoch_train_loss = 0.0 # Approximate training loss for the epoch
        train_steps = 0

        # Progress bar for visual feedback
        model.train() # Make sure model is in training mode for this epoch
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1} Training", leave=False)
        for batch in progress_bar:
            # Move data to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            token_type_ids = batch.get('token_type_ids')
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            # Forward pass
            logits = model(input_ids, attention_mask, token_type_ids)
            # Calculate loss
            loss = criterion(logits, labels)

            # --- Backpropagation and Optimization ---
            optimizer.zero_grad() # Clear previous gradients
            loss.backward()       # Compute gradients for the current batch
            optimizer.step()      # Update model parameters based on gradients

            epoch_train_loss += loss.item()
            train_steps += 1
            # Update progress bar postfix (optional)
            progress_bar.set_postfix({'loss': loss.item()})

        progress_bar.close()

        # --- Post-Epoch Evaluation (Now includes F1) ---
        # Note: Evaluating on the full training set can be time-consuming.
        train_loss_eval, train_acc, train_f1_real, train_f1_false = evaluate_with_loss_and_f1(model, train_loader, device, criterion)
        valid_loss, valid_acc, valid_f1_real, valid_f1_false = evaluate_with_loss_and_f1(model, valid_loader, device, criterion)

        # Record history (Added F1)
        history['train_loss'].append(train_loss_eval)
        history['train_acc'].append(train_acc)
        history['train_f1_real'].append(train_f1_real)
        history['train_f1_false'].append(train_f1_false)
        history['valid_loss'].append(valid_loss)
        history['valid_acc'].append(valid_acc)
        history['valid_f1_real'].append(valid_f1_real)
        history['valid_f1_false'].append(valid_f1_false)

        # --- Updated Print Statement ---
        print(f"  [Evaluation Results]")
        print(f"  Train Loss: {train_loss_eval:.4f} | Acc: {train_acc:.4f} | F1(Real): {train_f1_real:.4f} | F1(False): {train_f1_false:.4f}")
        print(f"  Valid Loss: {valid_loss:.4f} | Acc: {valid_acc:.4f} | F1(Real): {valid_f1_real:.4f} | F1(False): {valid_f1_false:.4f}")

        # --- Early Stopping Check (Based on Validation Accuracy) and Save Model ---
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            epochs_no_improve = 0 # <--- Reset counter
            print(f"  New best model found! Validation accuracy: {best_valid_acc:.4f}")
            # Save model state dictionary and tokenizer configuration
            model_save_path = os.path.join(output_dir, "best_model.pt")
            torch.save(model.state_dict(), model_save_path)
            tokenizer.save_pretrained(output_dir) # Saves tokenizer files
            print(f"  Best model saved to: {model_save_path}")
        else:
            epochs_no_improve += 1 # <--- Increment counter
            print(f"  Validation accuracy did not improve. ({epochs_no_improve}/{early_stopping_patience} epochs without improvement). Best Acc: {best_valid_acc:.4f}")
            # Check if patience is exhausted
            if early_stopping_patience > 0 and epochs_no_improve >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs (based on validation accuracy).")
                early_stop_triggered = True # <--- Mark early stopping
                break # <--- Break out of the training loop

    # --- Training Finished ---
    if not early_stop_triggered:
        print(f"\nTraining completed after {epochs} epochs.")
    else:
         print("\nTraining stopped early.")

    # --- Plot and Save Curves (plot based on actual number of epochs run) ---
    epochs_run = len(history['train_loss']) # Actual number of epochs completed
    if epochs_run > 0:
        # Loss Curve
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1) # Create subplot for Loss
        plt.plot(range(1, epochs_run + 1), history['train_loss'], marker='o', linestyle='-', label='Train Loss')
        plt.plot(range(1, epochs_run + 1), history['valid_loss'], marker='x', linestyle='--', label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)

        # Accuracy Curve
        plt.subplot(2, 2, 2) # Create subplot for Accuracy
        plt.plot(range(1, epochs_run + 1), history['train_acc'], marker='o', linestyle='-', label='Train Accuracy')
        plt.plot(range(1, epochs_run + 1), history['valid_acc'], marker='x', linestyle='--', label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        plt.grid(True)

        # F1 Real Curve
        plt.subplot(2, 2, 3) # Create subplot for F1 Real
        plt.plot(range(1, epochs_run + 1), history['train_f1_real'], marker='o', linestyle='-', label='Train F1 (Real=1)')
        plt.plot(range(1, epochs_run + 1), history['valid_f1_real'], marker='x', linestyle='--', label='Validation F1 (Real=1)')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.title('Training and Validation F1 (Real=1)')
        plt.legend()
        plt.grid(True)

        # F1 False Curve
        plt.subplot(2, 2, 4) # Create subplot for F1 False
        plt.plot(range(1, epochs_run + 1), history['train_f1_false'], marker='o', linestyle='-', label='Train F1 (False=0)')
        plt.plot(range(1, epochs_run + 1), history['valid_f1_false'], marker='x', linestyle='--', label='Validation F1 (False=0)')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.title('Training and Validation F1 (False=0)')
        plt.legend()
        plt.grid(True)

        plt.tight_layout() # Adjust layout to prevent overlap
        plot_path = os.path.join(output_dir, 'training_curves.png')
        plt.savefig(plot_path)
        print(f"Training curves saved to: {plot_path}")
        plt.close() # Close the plot to free memory
    else:
        print("No epochs were completed, skipping plotting.")

    # --- Final Report ---
    # The best model was already saved when improvement occurred (based on validation accuracy).
    # Report the best validation accuracy achieved.
    print(f"\nBest validation accuracy achieved during training: {best_valid_acc:.4f}")

    return best_valid_acc # Return the best validation accuracy found

# --- 6. Testing Function (Updated to calculate and return F1) ---
@torch.no_grad() # No need for gradients during testing
def test_model(test_file: str,
               model_dir: str, # Directory containing saved model and tokenizer
               pretrained_model_name_for_init: str = 'bert-base-uncased', # Used to initialize model architecture
               batch_size: int = 32,
               max_length: int = 512,
               dropout_prob_for_init: float = 0.3) -> Tuple[float, float, float]: # --- Updated return type ---
    """
    Loads the best saved model and tokenizer, evaluates it on the test set,
    and returns accuracy, F1(Real=1), and F1(False=0).

    Args:
        test_file (str): Path to the test data file (.jsonl).
        model_dir (str): Directory containing the saved model ('best_model.pt') and tokenizer config.
        pretrained_model_name_for_init (str): The pre-trained model name used when initializing MyBertClassifier.
                                               Should match the one used during training.
        batch_size (int): Batch size for testing.
        max_length (int): Maximum sequence length used by the tokenizer.
        dropout_prob_for_init (float): The dropout probability used when the model was trained and saved.
                                        Needed to correctly initialize the model structure before loading weights.

    Returns:
        Tuple[float, float, float]: Test accuracy, Test F1(Real=1), Test F1(False=0).
    """
    # Automatically select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- Starting Testing ---")
    print(f"Using device: {device}")

    # Load the saved tokenizer
    print(f"Loading tokenizer from '{model_dir}'...")
    try:
        tokenizer = BertTokenizer.from_pretrained(model_dir)
    except Exception as e:
        print(f"Failed to load tokenizer from {model_dir}: {e}")
        print(f"Attempting to load fallback tokenizer '{pretrained_model_name_for_init}'...")
        # Fallback: load the original pretrained tokenizer (might cause mismatch if vocab was extended)
        tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_for_init)

    # Create test Dataset and DataLoader
    print(f"Loading test data: {test_file}")
    try:
        test_dataset = QADataset(test_file, tokenizer, max_length=max_length)
    except FileNotFoundError:
        print(f"Error: Test file not found at {test_file}")
        return 0.0, 0.0, 0.0 # Return zeros if file not found
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    print(f"Test samples: {len(test_dataset)}")

    # Initialize model structure (using the same dropout as during training)
    model = MyBertClassifier(pretrained_model=pretrained_model_name_for_init, dropout_prob=dropout_prob_for_init)

    # Load the saved model weights (best model from training)
    model_path = os.path.join(model_dir, "best_model.pt") # Assume loading the best model
    print(f"Loading model weights from '{model_path}'...")
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return 0.0, 0.0, 0.0 # Return zeros if model not found

    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval() # Set model to evaluation mode
    except Exception as e:
        print(f"Failed to load model weights: {e}")
        return 0.0, 0.0, 0.0 # Return zeros on loading error

    # --- Evaluate model on the test set and calculate metrics ---
    print("Evaluating model on the test set...")
    all_labels = []
    all_predictions = []
    correct_predictions = 0
    total_samples = 0

    for batch in tqdm(test_loader, desc="Testing"):
        # Move batch data to the designated device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        token_type_ids = batch.get('token_type_ids') # Use .get to handle None safely
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)

        # Get model predictions (logits)
        logits = model(input_ids, attention_mask, token_type_ids)
        # Find the class with the highest probability as the prediction
        predictions = torch.argmax(logits, dim=1)

        # Accumulate accuracy counts
        correct_predictions += (predictions == labels).sum().item()
        total_samples += labels.size(0)

        # Collect labels and predictions for F1 calculation
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predictions.cpu().numpy())

    # Calculate final accuracy
    test_accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0

    # Calculate F1 scores
    test_f1_real = 0.0
    test_f1_false = 0.0
    if total_samples > 0:
        precision, recall, fscore, support = precision_recall_fscore_support(
            all_labels,
            all_predictions,
            average=None,
            labels=[0, 1],
            zero_division=0
        )
        if len(fscore) >= 2:
            test_f1_false = fscore[0] # F1 for class 0 (False)
            test_f1_real = fscore[1]  # F1 for class 1 (Real)
        # Handle edge case (less likely with labels=[0,1])
        elif len(fscore) == 1:
             present_labels = set(all_labels) | set(all_predictions)
             if 0 in present_labels and 1 not in present_labels: test_f1_false = fscore[0]
             elif 1 in present_labels and 0 not in present_labels: test_f1_real = fscore[0]

    print(f"\nTesting complete!")
    print(f"Test Set Accuracy: {test_accuracy:.4f} ({test_accuracy:.2%})")
    print(f"Test Set F1 Score (Real=1): {test_f1_real:.4f}")
    print(f"Test Set F1 Score (False=0): {test_f1_false:.4f}")

    return test_accuracy, test_f1_real, test_f1_false # --- Updated return values ---


# --- 7. Main Execution Block (Updated for F1 reporting) ---
if __name__ == "__main__":
    # --- Configuration Parameters ---
    TRAIN_FILE = 'ARG_dataset_jsonl/zh/rational_expert_ch_train.json'    # Path to the training set file
    VALID_FILE = 'ARG_dataset_jsonl/zh/val.jsonl'       # Path to the validation set file (using test_R1 here as validation)
    TEST_FILE  = 'ARG_dataset_jsonl/zh/test.jsonl'            # Path to the test set file (using test_R1 here as test)

    PRETRAINED_MODEL = 'bert-base-uncased' # Pre-trained BERT model to use
    OUTPUT_DIR = 'base_model_save'      # Output directory for model, tokenizer, plots
    LEARNING_RATE = 3e-5                   # Learning rate
    NUM_EPOCHS = 20                        # *Maximum* number of training epochs
    BATCH_SIZE_TRAIN = 64                  # Batch size for training
    BATCH_SIZE_EVAL = 32                   # Batch size for evaluation/testing (can be different)
    MAX_SEQ_LENGTH = 256                   # Maximum sequence length for tokenizer
    DROPOUT_PROB = 0.3                     # Dropout probability used in the model
    WEIGHT_DECAY = 0.01                    # Weight decay for AdamW optimizer
    EARLY_STOPPING_PATIENCE = 4            # Patience for early stopping based on validation accuracy

    # --- Execute Training ---
    print("="*30)
    print("Starting Model Training Process")
    print(f"Max Epochs: {NUM_EPOCHS}, Early Stopping Patience: {EARLY_STOPPING_PATIENCE}")
    print("="*30)
    best_validation_accuracy = train_model(
        train_file=TRAIN_FILE,
        valid_file=VALID_FILE,
        pretrained_model=PRETRAINED_MODEL,
        lr=LEARNING_RATE,
        epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE_TRAIN,
        max_length=MAX_SEQ_LENGTH,
        output_dir=OUTPUT_DIR,
        dropout_prob=DROPOUT_PROB,
        weight_decay=WEIGHT_DECAY,
        early_stopping_patience=EARLY_STOPPING_PATIENCE
    )
    print(f"\nTraining finished. Best validation accuracy achieved: {best_validation_accuracy:.4f}")

    # --- Execute Testing ---
    # Testing now uses the best model saved during training (based on validation accuracy)
    print("\n" + "="*30)
    print("Starting Model Testing Process (using the best saved model)")
    print("="*30)
    # --- Updated to capture F1 scores ---
    final_test_accuracy, final_test_f1_real, final_test_f1_false = test_model(
        test_file=TEST_FILE,
        model_dir=OUTPUT_DIR, # Load from the training output directory
        pretrained_model_name_for_init=PRETRAINED_MODEL,
        batch_size=BATCH_SIZE_EVAL,
        max_length=MAX_SEQ_LENGTH,
        dropout_prob_for_init=DROPOUT_PROB
    )
    # --- Updated final print statement ---
    print(f"\nTesting finished. Final Test Results:")
    print(f"  Accuracy: {final_test_accuracy:.4f}")
    print(f"  F1 Score (Real=1): {final_test_f1_real:.4f}")
    print(f"  F1 Score (False=0): {final_test_f1_false:.4f}")

    print("\nScript execution completed.")