import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
import pandas as pd

# Step 1: Load the Dataset
df = pd.read_csv('DNA_Fake_Dataset.csv')  # Load your DNA dataset
print("Dataset loaded successfully!")

# Display the first few rows of the dataset
print(df.head())

# Step 2: Preprocess the Data
# Assuming 'Sequence' is the DNA sequence and 'Mutation_Type' is the label
sequences = df['Sequence'].tolist()
labels = df['Mutation_Type'].tolist()

# Label encode the labels (Mutation_Type)
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Step 3: Prepare Dataset for Model
# Use Hugging Face's datasets library to work with the data
dataset = pd.DataFrame({'sequence': sequences, 'label': labels_encoded})

# Convert DataFrame to Hugging Face Dataset format
dataset = Dataset.from_pandas(dataset)

# Step 4: Tokenization
# Use pretrained BERT tokenizer to tokenize the DNA sequences
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_function(examples):
    return tokenizer(examples['sequence'], padding="max_length", truncation=True, max_length=512)

# Apply the tokenizer to the dataset
dataset = dataset.map(tokenize_function, batched=True)

# Step 5: Split the dataset into train and test sets
train_dataset, test_dataset = dataset.train_test_split(test_size=0.2).values()

# Step 6: Load Pretrained BERT Model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_encoder.classes_))

# Step 7: Check for CUDA (GPU) availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)  # Move model to the selected device (GPU or CPU)
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")

# Step 8: Training Setup
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # number of training epochs
    per_device_train_batch_size=8,   # batch size for training
    per_device_eval_batch_size=16,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
    evaluation_strategy="epoch",     # evaluate each epoch
    save_strategy="epoch",           # save the model each epoch
)

# Step 9: Initialize the Trainer
trainer = Trainer(
    model=model,                         # the pretrained model
    args=training_args,                  # training arguments
    train_dataset=train_dataset,         # training dataset
    eval_dataset=test_dataset,           # evaluation dataset
    tokenizer=tokenizer                  # tokenizer
)

# Step 10: Train the Model
trainer.train()

# Step 10.1: Save the Trained Model and Tokenizer
model.save_pretrained('./main1')
tokenizer.save_pretrained('./main1')
print("Model and tokenizer saved successfully!")

# Step 11: Evaluate the Model
results = trainer.evaluate()
print(f"Test results: {results}")


# Step 12: Predict with the Trained Model
def predict_with_features(sequence):
    # Tokenize the input sequence
    inputs = tokenizer(sequence, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    with torch.no_grad():
        output = model(**inputs)
        logits = output.logits
        predicted_class = torch.argmax(logits, dim=-1).item()  # Get the predicted class
        predicted_label = label_encoder.inverse_transform([predicted_class])[0]  # Convert to original label
        
        # Check if the sequence exists in the DataFrame
        if sequence not in df['Sequence'].values:
            return {"Error": "Sequence not found in the dataset."}
        
        # Fetch the sequence features based on the predicted label
        sequence_index = df[df['Sequence'] == sequence].index[0]
        features = df.iloc[sequence_index].to_dict()
        
        # Return the predicted mutation type and the features
        return {'Predicted_Mutation_Type': predicted_label, **features}

# Example usage: Predict the features of a DNA sequence
new_sequence = "ATGCGTACGTAAGCGT"  # Example DNA sequence
output = predict_with_features(new_sequence)
print(output)  # This will output all features including mutation type, gene, etc.
