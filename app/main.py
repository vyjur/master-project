from flask import Flask, render_template, jsonify, request
from werkzeug.utils import secure_filename
from bs4 import BeautifulSoup
from datetime import datetime
import os
import uuid

from pipeline.setup import Pipeline

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'app','uploads')
pipeline = Pipeline()

@app.route("/", methods=("GET", "POST"))
def index():

    if request.method == "POST":
        if request.form["type"] == "upload":
            if 'files' in request.files:
                files = request.files.getlist('files')
                uploaded_files = []
                batch_name = str(uuid.uuid4()) 
                for i, file in enumerate(files):
                    if file:
                        filename = batch_name + "+" + str(i) + "+" + secure_filename(file.filename)
                        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                        uploaded_files.append(filename)
                pipeline.init(False, files)
        elif request.form["type"] == "add":
            pipeline.add(request.form['symptom'])
        elif request.form["type"] == "predict":
            pipeline.predict(request.form['id'])
        
    with open("app/templates/simple_network.html", "r", encoding="utf-8") as file:
        html_content = file.read()

    soup = BeautifulSoup(html_content, "html.parser")

    script_tag = soup.find_all("script")

    # Modify the JavaScript content inside the <script> tag
    extra_js_content = """
    network.on('click', async function(properties) {

                    var clickedNodeId = properties.nodes[0]; // Get the ID of the clicked node

                    if (clickedNodeId == undefined) {
                        return
                    }

                    var result = await fetch("/node/"+clickedNodeId,
                    {
                        headers: {
                        'Accept': 'application/json',
                        'Content-Type': 'application/json'
                        },
                        method: "GET",
                    })

                    var id = await result.json()
                    document.getElementById('id').innerHTML = id
                    document.getElementById('predictId').value = id
                })
    """

    script_tag[-1].string += extra_js_content

    with open("app/templates/simple_network2.html", "w", encoding="utf-8") as file:
        file.write(str(soup))

    return render_template("index.html")


@app.route("/node/<int:id>")
def node(id):
    return jsonify(id)


if __name__ == "__main__":
    app.run(debug=True)
