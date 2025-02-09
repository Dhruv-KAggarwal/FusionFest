import hashlib
import uuid
import os

# Create a folder named "HASHES" if it doesn't exist
HASHES_FOLDER = "HASHES"
os.makedirs(HASHES_FOLDER, exist_ok=True)

def hash_sha256(sequence):
    """Hashes the input sequence using SHA-256"""
    return hashlib.sha256(sequence.encode()).hexdigest()

def save_to_fasta(sequence, hashed_value, filename="hashed_output.fasta"):
    """Saves the hashed sequence in FASTA format inside the HASHES folder"""
    unique_id = str(uuid.uuid4())[:8]  # Generate a short unique identifier
    filepath = os.path.join(HASHES_FOLDER, filename)  # Save in HASHES folder

    with open(filepath, "a") as fasta_file:
        fasta_file.write(f">{unique_id}_hash\n")  # FASTA header
        fasta_file.write(f"{hashed_value}\n\n")  # Hashed sequence

if __name__ == "__main__":
    sequence = input("Enter the alphabet sequence: ")  # User input
    hashed_value = hash_sha256(sequence)  # Hash the input
    
    save_to_fasta(sequence, hashed_value)  # Save hash in FASTA format
    print(f"Hash saved in {HASHES_FOLDER}/hashed_output.fasta\n")
