from flask import Flask, request, jsonify
from transformers import BertForSequenceClassification, BertTokenizer
import torch
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the saved model and tokenizer
MODEL_PATH = "saved_model"  # Path to your saved model folder
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model.eval()  # Set the model to evaluation mode

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        # Ensure the input contains the DNA sequence
        if "sequence" not in data:
            return jsonify({"error": "No DNA sequence provided"}), 400
        
        sequence = data["sequence"]
        if not isinstance(sequence, str):
            return jsonify({"error": "DNA sequence must be a string"}), 400

        # Tokenize the input sequence
        inputs = tokenizer(sequence, return_tensors="pt", truncation=True, padding=True, max_length=512)
        
        # Get model predictions
        with torch.no_grad():
            outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1).tolist()

        return jsonify({"predictions": probabilities})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
