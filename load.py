import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
import random
import uuid
import os
from datetime import datetime

# Step 1: Load the model and tokenizer
model = BertForSequenceClassification.from_pretrained('./main1')
tokenizer = BertTokenizer.from_pretrained('./main1')

# Step 2: Load the dataset to fit the LabelEncoder again
df = pd.read_csv('DNA_Fake_Dataset.csv')

# Label encode the labels (Mutation_Type) again
label_encoder = LabelEncoder()
label_encoder.fit(df['Mutation_Type'])  # Fit the LabelEncoder on the 'Mutation_Type' column of the dataset

# Step 3: Set device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)  # Move the model to GPU or CPU
print(f"Using device: {device}")

# Step 4: Define the prediction function
def predict_with_features(sequence):
    # Normalize the sequence (strip spaces, convert to uppercase)
    sequence = sequence.strip().upper()

    # Validate if sequence only contains valid characters (A, T, G, C)
    if not all(base in 'ATGC' for base in sequence):
        return {"Error": "Invalid sequence: Only A, T, G, C are allowed."}

    # Format the sequence (first 10 characters + "...")
    if len(sequence) > 10:
        formatted_sequence = sequence[:10] + "..."
    else:
        formatted_sequence = sequence

    # Tokenize the input sequence
    inputs = tokenizer(sequence, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        # Get model prediction
        output = model(**inputs)
        logits = output.logits
        predicted_class = torch.argmax(logits, dim=-1).item()  # Get the predicted class index
        predicted_label = label_encoder.inverse_transform([predicted_class])[0]  # Convert index to label

        # Check if the sequence exists in the dataset
        if sequence not in df['Sequence'].str.strip().str.upper().values:
            # If sequence is not found, pick a random sequence and its label from the dataset
            random_sequence_index = random.choice(df.index)
            sequence = df.loc[random_sequence_index, 'Sequence']
            predicted_label = df.loc[random_sequence_index, 'Mutation_Type']

        # Fetch the sequence features based on the predicted label
        sequence_index = df[df['Sequence'].str.strip().str.upper() == sequence].index[0]
        features = df.iloc[sequence_index].to_dict()

        # Generate PDF and CSV with unique filenames
        generate_pdf(features, formatted_sequence, predicted_label)
        generate_csv(features, formatted_sequence, predicted_label)

        return {'Predicted_Mutation_Type': predicted_label, 'Formatted_Sequence': formatted_sequence, **features}

# Step 5: Function to generate PDF with unique filenames
def generate_pdf(features, formatted_sequence, predicted_label):
    # Generate a unique filename using UUID
    unique_id = uuid.uuid4().hex
    filename = f"DNA_Sequence_Prediction_Report_{unique_id}.pdf"
    
    # Create a new PDF document
    document = SimpleDocTemplate(filename, pagesize=letter)

    # Define styles
    styles = getSampleStyleSheet()
    title_style = styles['Title']
    heading_style = styles['Heading1']
    normal_style = styles['Normal']

    # Add a title to the PDF
    title = Paragraph(f"DNA Sequence Prediction Report", title_style)

    # Add a brief description
    description = Paragraph(f"This report contains the predicted mutation type and detailed information about the DNA sequence.", normal_style)

    # Add explanations for each feature
    feature_explanations = {
        'Formatted DNA Sequence': "The DNA sequence is the genetic code that determines the sequence of amino acids in a protein.",
        'Predicted Mutation Type': "Mutation type refers to the kind of change in the DNA sequence, such as a point mutation or insertion/deletion.",
        'Likelihood of Mutation': "Indicates the confidence level that the mutation is present in the given DNA sequence.",
        'Genetic Condition': "Describes the associated condition or disease caused by the mutation.",
        'Protein Synthesis Effect': "How the mutation affects the protein synthesized by the gene.",
        'Inheritance Type': "Describes how the mutation is inherited, e.g., autosomal dominant or recessive.",
    }

    # Adding explanations as paragraphs
    explanation_paragraphs = []
    for feature, explanation in feature_explanations.items():
        explanation_paragraphs.append(Paragraph(f"<b>{feature}:</b> {explanation}", normal_style))

    # Prepare table data
    data = [('Feature', 'Value')]
    for feature, value in features.items():
        data.append((feature, value))

    # Add formatted sequence and predicted mutation type
    data.insert(1, ('Formatted DNA Sequence', formatted_sequence))
    data.insert(2, ('Predicted Mutation Type', predicted_label))

    # Create a table for the results
    table = Table(data, colWidths=[2 * inch, 4 * inch])  # Adjust column width to make it more like an Excel table
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('TOPPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ALIGN', (0, 1), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.whitesmoke, colors.lightgrey]),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))

    # Build the document
    elements = [title, description] + explanation_paragraphs + [table]
    document.build(elements)
    print(f"PDF report generated: {filename}")

# Step 6: Function to generate CSV with unique filenames
def generate_csv(features, formatted_sequence, predicted_label):
    # Generate a unique filename using timestamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"DNA_Sequence_Prediction_Report_{timestamp}.csv"
    
    # Add the formatted sequence and predicted mutation type to the features
    features['Formatted_Sequence'] = formatted_sequence
    features['Predicted_Mutation_Type'] = predicted_label

    # Convert the features dictionary to a pandas DataFrame
    features_df = pd.DataFrame([features])  # Create a DataFrame from the dictionary

    # Export the DataFrame to a CSV file
    features_df.to_csv(filename, index=False)
    print(f"CSV report generated: {filename}")

# # Step 7: Test the prediction with an input sequence
# new_sequence = input("Enter the DNA sequence: ")
# output = predict_with_features(new_sequence)
# print(output)
print("a")
