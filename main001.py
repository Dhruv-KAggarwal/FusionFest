# import numpy as np

# def dna_to_onehot(sequence):
#     # Define the possible nucleotides
#     nucleotides = ['A', 'C', 'G', 'T']
#     encoding = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1]}
    
#     onehot_encoded = np.array([encoding[nucleotide] for nucleotide in sequence if nucleotide in encoding])
#     return onehot_encoded

# # Example DNA sequence
# sequence = "TAACCCTAACCCTAACCCTA"
# onehot_sequence = dna_to_onehot(sequence)
# print(onehot_sequence)


# import torch
# from torch.utils.data import Dataset, DataLoader

# class DNADataset(Dataset):
#     def __init__(self, sequences, labels):
#         self.sequences = sequences
#         self.labels = labels

#     def __len__(self):
#         return len(self.sequences)

#     def __getitem__(self, idx):
#         seq = self.sequences[idx]
#         label = self.labels[idx]
#         return torch.tensor(seq, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# # Example of sequences and corresponding labels (replace with your data)
# sequences = [dna_to_onehot(seq) for seq in ["TAACCCTAACCCTAACCCTA", "CGTACGTA"]]
# labels = [1, 0]  # Replace with your labels (binary or multi-class)

# # Create Dataset and DataLoader
# dataset = DNADataset(sequences, labels)
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


# import torch.nn as nn

# class DNA_LSTM_Model(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(DNA_LSTM_Model, self).__init__()
#         self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
#         self.fc = nn.Linear(hidden_size, output_size)
    
#     def forward(self, x):
#         lstm_out, (h_n, c_n) = self.lstm(x)
#         last_hidden = h_n[-1]  # Take the last hidden state
#         out = self.fc(last_hidden)
#         return out

# # Initialize model
# input_size = 4  # For one-hot encoding, each nucleotide has 4 possible values
# hidden_size = 128  # You can experiment with this
# output_size = 2  # Binary classification (adjust based on your needs)

# model = DNA_LSTM_Model(input_size, hidden_size, output_size)



# # Define loss function and optimizer
# criterion = nn.CrossEntropyLoss()  # For classification tasks
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# # Training Loop
# num_epochs = 10
# for epoch in range(num_epochs):
#     for batch_idx, (data, target) in enumerate(dataloader):
#         optimizer.zero_grad()  # Reset the gradients
#         output = model(data)  # Forward pass
#         loss = criterion(output, target)  # Calculate loss
#         loss.backward()  # Backpropagate the gradients
#         optimizer.step()  # Update model parameters

#         if batch_idx % 10 == 0:
#             print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}')



import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Step 1: Data Preprocessing (One-Hot Encoding)
def one_hot_encode(sequence, seq_length):
    encoding = {'A': [1, 0, 0, 0], 'T': [0, 1, 0, 0], 'C': [0, 0, 1, 0], 'G': [0, 0, 0, 1]}
    sequence_encoded = np.zeros((seq_length, 4))
    for i, nucleotide in enumerate(sequence):
        if i >= seq_length:  # truncate the sequence if it's longer than seq_length
            break
        sequence_encoded[i] = encoding.get(nucleotide, [0, 0, 0, 0])  # if the nucleotide is invalid, assign zero vector
    return sequence_encoded

# Example data: List of DNA sequences
sequences = ["TAACCCTA", "ACCCTAACC", "CCTAAACC"]  # Example list of sequences
labels = [0, 1, 0]  # Example labels (binary classification)

# Convert sequences to one-hot encoded numpy arrays
seq_length = 9  # Define sequence length (pad/truncate sequences to this length)
encoded_sequences = np.array([one_hot_encode(seq, seq_length) for seq in sequences])

# Step 2: Dataset Class for PyTorch
class DNASequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(encoded_sequences, labels, test_size=0.2, random_state=42)

# Create DataLoader for training and validation
train_dataset = DNASequenceDataset(X_train, y_train)
val_dataset = DNASequenceDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

# Step 3: Define the Model (CNN-based)
class DNASequenceModel(nn.Module):
    def __init__(self):
        super(DNASequenceModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=64, kernel_size=3, padding=1)  # Convolutional layer
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)  # Max pooling
        self.fc1 = nn.Linear(64 * (seq_length // 2), 128)  # Fully connected layer
        self.fc2 = nn.Linear(128, 2)  # Output layer (binary classification)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = x.permute(0, 2, 1)  # Change input shape to (batch_size, channels, seq_length)
        x = self.relu(self.conv1(x))  # Apply convolution and ReLU
        x = self.pool(x)  # Apply max pooling
        x = x.view(x.size(0), -1)  # Flatten the output for fully connected layers
        x = self.relu(self.fc1(x))  # Apply first fully connected layer
        x = self.fc2(x)  # Apply second fully connected layer (output)
        return x

# Step 4: Training the Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DNASequenceModel().to(device)
criterion = nn.CrossEntropyLoss()  # Suitable for classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()  # Zero the gradients
        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights
        
        running_loss += loss.item()
        
        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_predictions += labels.size(0)
    
    train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct_predictions / total_predictions
    
    # Validation
    model.eval()
    val_loss = 0.0
    val_correct_predictions = 0
    val_total_predictions = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            val_correct_predictions += (predicted == labels).sum().item()
            val_total_predictions += labels.size(0)
    
    val_loss = val_loss / len(val_loader)
    val_accuracy = 100 * val_correct_predictions / val_total_predictions
    
    # Print training and validation stats
    print(f"Epoch [{epoch+1}/{num_epochs}]")
    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
    print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

print("Training finished!")
# Save the model
torch.save(model.state_dict(), 'dna_sequence_model.pth')

# To load the model later
model = DNASequenceModel().to(device)  # Initialize the model
# To load the model weights with 'weights_only=True'
model.load_state_dict(torch.load('dna_sequence_model.pth', weights_only=True))  # Load the saved weights safely
model.eval()  # Set to evaluation mode
