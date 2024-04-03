import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
import torch
from torch.optim import AdamW  
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os

# Load pre-trained BERT tokenizer and model for sequence classification
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Load the data from the CSV file
data = pd.read_csv('amp_dataset.csv')

# Split the data into features (sequences) and labels
sequences = data['sequence'].values
labels = data['label'].values

# Split the data into training, validation, and test sets
train_sequences, temp_sequences, train_labels, temp_labels = train_test_split(
    sequences, labels, test_size=0.2, random_state=42, stratify=labels)

val_sequences, test_sequences, val_labels, test_labels = train_test_split(
    temp_sequences, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels)

# Tokenize the input sequences with reduced batch size
batch_size = 256

train_encodings = tokenizer(train_sequences.tolist(), truncation=True, padding=True, return_tensors='pt', max_length=512)
val_encodings = tokenizer(val_sequences.tolist(), truncation=True, padding=True, return_tensors='pt', max_length=512)
test_encodings = tokenizer(test_sequences.tolist(), truncation=True, padding=True, return_tensors='pt', max_length=512)

# Convert labels to PyTorch tensors
train_labels = torch.tensor(train_labels)
val_labels = torch.tensor(val_labels)
test_labels = torch.tensor(test_labels)

# Create DataLoader objects with reduced batch size
train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], train_labels)
val_dataset = TensorDataset(val_encodings['input_ids'], val_encodings['attention_mask'], val_labels)
test_dataset = TensorDataset(test_encodings['input_ids'], test_encodings['attention_mask'], test_labels)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Fine-tuning parameters
optimizer = AdamW(model.parameters(), lr=5e-5)

# Training loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

model_file_path = 'Models/model_file.bin'

if not os.path.exists(model_file_path):
    # The model file does not exist, proceed with training
    print("Model file does not exist. Starting training.")


    # Lists to store losses
    train_losses = []
    val_losses = []
        
    # Training
    num_epochs = 3
    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0.0
        # Wrap train_loader with tqdm for a progress bar
        train_progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} Training")
        for batch in train_progress_bar:
            input_ids, attention_mask, labels = batch
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
            # Optionally update the progress bar with each batch's loss
            train_progress_bar.set_postfix({'batch_loss': loss.item()})

        epoch_train_loss /= len(train_loader)
        train_losses.append(epoch_train_loss)

        model.eval()
        epoch_val_loss = 0.0
        val_predictions = []
        val_targets = []
        # Wrap val_loader with tqdm for a progress bar
        val_progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} Validation")
        for batch in val_progress_bar:
            input_ids, attention_mask, labels = batch
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            
            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                epoch_val_loss += loss.item()
                
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                val_predictions.extend(preds)
                val_targets.extend(labels.cpu().numpy())

        epoch_val_loss /= len(val_loader)
        val_losses.append(epoch_val_loss)

        # Calculate metrics for validation
        val_accuracy = accuracy_score(val_targets, val_predictions)
        val_precision = precision_score(val_targets, val_predictions, zero_division=0)  # Add zero_division parameter
        val_recall = recall_score(val_targets, val_predictions)
        val_f1_score = f1_score(val_targets, val_predictions)

        # Convert val_targets to a NumPy array
        val_targets_np = np.array(val_targets)

        print("Shape of val_targets:", val_targets_np.shape)
        print("Shape of predicted probabilities:", logits[:, 1].cpu().numpy().shape)

    if len(logits) > 1 and len(val_targets_np) == len(logits[:, 1].cpu().numpy()):  # Check if there are predicted samples
        val_roc_auc = roc_auc_score(val_targets_np, logits[:, 1].cpu().numpy())
    else:
        val_roc_auc = 0.5  # Set default value if no predicted samples or inconsistent numbers of samples

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_train_loss}, Validation Loss: {epoch_val_loss}")
    print(f"Validation Accuracy: {val_accuracy}, Precision: {val_precision}, Recall: {val_recall}, F1 Score: {val_f1_score}, ROC AUC: {val_roc_auc}")

    # Plotting the losses
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.show()
    plt.savefig('Training plot.png')


    # Saving the model
    torch.save(model.state_dict(), model_file_path)
else:
    # The model file exists, load the model
    print("Model file exists. Loading model.")
    
    # Load your model. This also depends on the framework you're using.
    # For PyTorch, loading would look something like this:
    model.load_state_dict(torch.load(model_file_path))
    
# Evaluation on test set
test_loss = 0
test_predictions = []
test_targets = []
accumulated_logits = []  # List to accumulate logits across batches
model.eval()

# Wrap test_loader with tqdm for a progress bar
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Testing"):
        input_ids, attention_mask, labels = batch
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
        
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        test_loss += loss.item()
        
        logits = outputs.logits
        accumulated_logits.extend(logits[:, 1].cpu().numpy())  # Accumulate logits
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        test_predictions.extend(preds)
        test_targets.extend(labels.cpu().numpy())

test_loss /= len(test_loader)

# Calculate metrics
test_accuracy = accuracy_score(test_targets, test_predictions)
test_precision = precision_score(test_targets, test_predictions, zero_division=0)
test_recall = recall_score(test_targets, test_predictions)
test_f1_score = f1_score(test_targets, test_predictions)

# Calculate ROC AUC score using accumulated logits
test_roc_auc = roc_auc_score(test_targets, accumulated_logits)

print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")
print(f"Test Precision: {test_precision}")
print(f"Test Recall: {test_recall}")
print(f"Test F1 Score: {test_f1_score}")
print(f"Test ROC AUC: {test_roc_auc}")

