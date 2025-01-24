from flask import Flask, render_template, request, send_file
import os
import load  # Ensure your `load.py` is in the same directory as `app.py`
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Folder to store the result files temporarily for download
RESULTS_FOLDER = "results"
UPLOAD_FOLDER = "uploads"

if not os.path.exists(RESULTS_FOLDER):
    os.makedirs(RESULTS_FOLDER)

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'fasta_file' not in request.files:
        return render_template('index.html', error="No file part in the request")

    file = request.files['fasta_file']

    if file.filename == '':
        return render_template('index.html', error="No selected file")

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Read the content of the FASTA file
        with open(file_path, 'r') as f:
            sequence = ''.join(line.strip() for line in f if not line.startswith('>'))

        # Call the function to make predictions (assuming you have it in load.py)
        result = load.predict_with_features(sequence)

        if 'Error' in result:
            return render_template('index.html', error=result['Error'])

        prediction = result['Predicted_Mutation_Type']
        formatted_sequence = result['Formatted_Sequence']
        features = {k: v for k, v in result.items() if k not in ['Predicted_Mutation_Type', 'Formatted_Sequence']}

        # Save the result to a text file for download
        result_file = os.path.join(RESULTS_FOLDER, "prediction_result.txt")
        with open(result_file, "w") as f:
            f.write(f"Predicted Mutation Type: {prediction}\n")
            f.write(f"Formatted Sequence: {formatted_sequence}\n")
            f.write("\nAdditional Features:\n")
            for feature, value in features.items():
                f.write(f"{feature}: {value}\n")

        # Render the results back to the index.html template with a download link
        return render_template('index.html', prediction=prediction, sequence=formatted_sequence, features=features, result_file=result_file)


@app.route('/download')
def download():
    file_path = request.args.get('file')
    return send_file(file_path, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)
