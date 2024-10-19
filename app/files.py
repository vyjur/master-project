from flask import Flask, request, jsonify, abort
import os

# HOW TO POST:
# curl -X POST -F "files=@/path/to/your/local/file1.txt" -F "files=@/path/to/your/local/file2.png" http://<your-ip-address>:5000/api/upload

app = Flask(__name__)

# Set the directory where files will be saved
UPLOAD_DIRECTORY = 'data/'  # Update this path

# Create the upload directory if it doesn't exist
if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)

@app.route('/api/upload', methods=['POST'])
def upload_files():
    # Check if the request contains files
    if 'files' not in request.files:
        return jsonify({"error": "No file part"}), 400

    files = request.files.getlist('files')  # Retrieve the list of files
    if not files:
        return jsonify({"error": "No selected files"}), 400

    uploaded_files = []
    for file in files:
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        # Save the file to the specified directory
        file_path = os.path.join(UPLOAD_DIRECTORY, file.filename)
        file.save(file_path)
        uploaded_files.append(file.filename)

    return jsonify({"message": "Files uploaded successfully", "files": uploaded_files}), 201

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)  # Listen on all interfaces
