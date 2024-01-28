from transformers import RobertaTokenizer, RobertaForSequenceClassification, RobertaConfig
from torch.utils.data import DataLoader, TensorDataset
import torch
from sklearn.model_selection import train_test_split
import pandas as pd
from sympy import sympify, simplify

# Step 1: Load and Preprocess Dataset
# The data us in jason Format. Select the FinQA or MathQA data set files as a input
dataset_filepath = '/datasets/raw_data/'
df = pd.read_csv(dataset_filepath)
df = df.dropna()  # Drop any rows with missing values

# Split the dataset into training and validation sets
train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

# Tokenize the datasets
tokenizer = RobertaTokenizer.from_pretrained('roberta-large')

train_tokenized = tokenizer(
    train_df['word_problem'].tolist(),
    max_length=128,
    padding='max_length',
    truncation=True,
    return_tensors='pt'
)

val_tokenized = tokenizer(
    val_df['word_problem'].tolist(),
    max_length=128,
    padding='max_length',
    truncation=True,
    return_tensors='pt'
)

train_labels = tokenizer(
    train_df['equation'].tolist(),
    max_length=128,
    padding='max_length',
    truncation=True,
    return_tensors='pt'
)['input_ids']

val_labels = tokenizer(
    val_df['equation'].tolist(),
    max_length=128,
    padding='max_length',
    truncation=True,
    return_tensors='pt'
)['input_ids']

# Step 2: Define Model Architecture
config = RobertaConfig.from_pretrained('roberta-large', num_labels=len(tokenizer.get_vocab()))
model = RobertaForSequenceClassification.from_pretrained('roberta-large', config=config)

# Step 3: Fine-Tune the Model
train_input_ids = train_tokenized['input_ids']
train_attention_mask = train_tokenized['attention_mask']
train_labels = train_labels

train_dataset = TensorDataset(train_input_ids, train_attention_mask, train_labels)

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    for batch in train_dataloader:
        inputs, masks, targets = batch
        outputs = model(inputs, attention_mask=masks, labels=targets)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# Step 4: Generate Equations and Extract Answers
model.eval()
val_input_ids = val_tokenized['input_ids']
val_attention_mask = val_tokenized['attention_mask']
val_labels = val_labels

val_dataset = TensorDataset(val_input_ids, val_attention_mask, val_labels)

val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)

total_correct = 0
total_samples = 0

with torch.no_grad():
    for val_batch in val_dataloader:
        val_inputs, val_masks, val_targets = val_batch
        val_outputs = model(val_inputs, attention_mask=val_masks, labels=val_targets)
        logits = val_outputs.logits
        predictions = torch.argmax(logits, dim=2)
        total_correct += torch.sum(predictions == val_targets).item()
        total_samples += val_targets.numel()

        # Extract answer using SymPy
        generated_equation = tokenizer.decode(predictions[0], skip_special_tokens=True)
        try:
            sympy_equation = sympify(generated_equation)
            correct_answer = simplify(sympy_equation.evalf())  # Evaluate the expression
            print(f"Generated Equation: {generated_equation}, Correct Answer: {correct_answer}")
        except:
            print(f"Failed to process equation: {generated_equation}")

accuracy = total_correct / total_samples
print(f"Validation Accuracy: {accuracy * 100:.2f}%")