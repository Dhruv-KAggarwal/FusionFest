from flask import Flask, render_template, request, send_file, redirect, url_for, flash
import os
from werkzeug.utils import secure_filename
from fpdf import FPDF  # For generating PDFs
import load  # Ensure your `load.py` is in the same directory as `app.py`

app = Flask(__name__)

# Configurations
app.secret_key = "your_secret_key"  # Replace with a secure key
RESULTS_FOLDER = "results"
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"fasta"}  # Allowed file extensions

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create necessary directories
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    """Check if the file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    sequence = None
    pdf_file_path = None

    # Check for file upload
    if 'fasta_file' in request.files and request.files['fasta_file'].filename:
        file = request.files['fasta_file']
        
        if not allowed_file(file.filename):
            flash(f"Invalid file type. Only {', '.join(ALLOWED_EXTENSIONS)} files are allowed.", "error")
            return redirect(url_for('home'))

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Extract sequence from file
        with open(file_path, 'r') as f:
            sequence = ''.join(line.strip() for line in f if not line.startswith('>'))

    # Check for direct sequence input
    elif 'sequence_input' in request.form:
        sequence = request.form['sequence_input'].strip()

    if not sequence:
        flash("No valid input provided.", "error")
        return redirect(url_for('home'))

    try:
        # Call the prediction function from `load.py`
        result = load.predict_with_features(sequence)

        if 'Error' in result:
            flash(result['Error'], "error")
            return redirect(url_for('home'))

        # Extract results
        prediction = result['Predicted_Mutation_Type']
        formatted_sequence = result['Formatted_Sequence']
        features = {k: v for k, v in result.items() if k not in ['Predicted_Mutation_Type', 'Formatted_Sequence']}

        # Generate PDF with results
        pdf_file_path = os.path.join(RESULTS_FOLDER, "prediction_result.pdf")
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="DNA Analysis Result", ln=True, align='C')
        pdf.ln(10)
        pdf.multi_cell(0, 10, f"Predicted Mutation Type: {prediction}")
        pdf.multi_cell(0, 10, f"Formatted Sequence:\n{formatted_sequence}")
        pdf.ln(5)
        pdf.cell(0, 10, txt="Additional Features:", ln=True)
        for feature, value in features.items():
            pdf.cell(0, 10, txt=f"{feature}: {value}", ln=True)

        pdf.output(pdf_file_path)

        flash("Prediction completed. You can download the results as a PDF.", "success")
        return render_template(
            'index.html',
            prediction=prediction,
            sequence=formatted_sequence,
            features=features,
            pdf_file=pdf_file_path
        )

    except Exception as e:
        flash(f"An error occurred: {str(e)}", "error")
        return redirect(url_for('home'))

@app.route('/download')
def download():
    file_path = request.args.get('file')
    if not file_path or not os.path.exists(file_path):
        flash("File not found for download.", "error")
        return redirect(url_for('home'))

    return send_file(file_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
