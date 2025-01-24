import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Preprocessing
def preprocess_data(csv_file):
    df = pd.read_csv(csv_file)

    # Check if 'Pathogenic_Score' column exists (replace with actual target column name if needed)
    target_column = 'Pathogenic_Score'  # Change this if you have a different target column
    if target_column not in df.columns:
        raise KeyError(f"Column '{target_column}' not found in the dataset!")

    # Handle missing or inconsistent data
    df = df.fillna(0)  # Replace NaN with 0 for simplicity

    # Label encoding for categorical columns
    label_encoders = {}
    categorical_columns = ["Mutation_Type", "Gene", "Ancestral_Region", "Genetic_Condition",
                           "Protein_Synthesis_Effect", "Inheritance_Type", "Phenotypic_Expression",
                           "Ethnic_Association", "Evolutionary_Clade", "Epigenetic_Modifications"]
    for column in categorical_columns:
        if column in df.columns and df[column].dtype == 'object':
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column].astype(str))
            label_encoders[column] = le

    # Scale numerical columns
    scaler = StandardScaler()
    numerical_columns = ["Length", "GC_Content", "Likelihood_of_Mutation", "Pathogenic_Score"]
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

    # Drop the 'Sequence' column, as we will not process it
    df = df.drop(columns=['Sequence'])
    
    return df, label_encoders, scaler

# Custom Dataset Class
class MutationDataset(Dataset):
    def __init__(self, data, target_column='Pathogenic_Score'):
        # Ensure the target column is present
        self.features = data.drop(columns=[target_column]).values.astype(np.float32)
        self.labels = data[target_column].values.astype(np.float32)  # Ensure this targets the correct column

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y

# Neural Network Model
class MutationPredictor(nn.Module):
    def __init__(self, input_size, output_size):
        super(MutationPredictor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        return self.model(x)

# Training function
def train_model(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for data, labels in dataloader:
        data, labels = data.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)

# Validation function
def validate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data, labels in dataloader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    return total_loss / len(dataloader)

# Main pipeline
def main():
    # Parameters
    csv_file = 'DNA_Fake_Dataset.csv'
    batch_size = 32
    epochs = 50
    learning_rate = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Preprocess data
    df, label_encoders, scaler = preprocess_data(csv_file)
    target_column = 'Pathogenic_Score'  # Use the correct target column name
    train_data, val_data = train_test_split(df, test_size=0.2, random_state=42)

    train_dataset = MutationDataset(train_data, target_column)
    val_dataset = MutationDataset(val_data, target_column)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model, optimizer, loss
    input_size = train_data.shape[1] - 1  # Exclude target column
    output_size = 1  # Predicting a single value for Target

    model = MutationPredictor(input_size, output_size).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # Training loop
    for epoch in range(epochs):
        train_loss = train_model(model, train_loader, optimizer, criterion, device)
        val_loss = validate_model(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

    # Save the model
    torch.save(model.state_dict(), "mutation_predictor.pth")
    print("Model saved as 'mutation_predictor.pth'!")

if __name__ == "__main__":
    main()
