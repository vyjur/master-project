from flask import (
    Flask,
    render_template,
    redirect,
    url_for,
    request,
    send_from_directory,
)
from pipeline.setup import Pipeline
import os
import pypdf


SAVE_PATH = "./app/templates/output/"


app = Flask(__name__)


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/upload", methods=["POST"])
def upload():
    # INFO: Need hardware resources for this to work!!!
    pipeline = Pipeline("./src/pipeline/config.ini")

    uploaded_file = request.files.get("file")

    if not uploaded_file or uploaded_file.filename == "":
        return "No file selected", 400

    filename = uploaded_file.filename
    filepath = os.path.join(SAVE_PATH, filename)  # type: ignore
    os.makedirs(SAVE_PATH, exist_ok=True)
    uploaded_file.save(filepath)

    reader = pypdf.PdfReader(filepath)
    documents = [page.extract_text() for page in reader.pages]
    os.makedirs(SAVE_PATH, exist_ok=True)
    pipeline.run(documents, save_path=SAVE_PATH, dct_cp=True)

    return redirect(url_for("home"))


@app.route("/save")
def save():
    # Update the path to a directory that Flask can serve files from
    output_dir = os.path.join(os.getcwd(), "app", "templates", "output")
    file_name = "final.csv"
    file_path = os.path.join(output_dir, file_name)
    # Check if the file exists
    if os.path.exists(file_path):
        # Use send_from_directory to safely send the file as an attachment
        return send_from_directory(output_dir, file_name, as_attachment=True)
    else:
        return "File not found", 404


if __name__ == "__main__":
    app.run(debug=True)

