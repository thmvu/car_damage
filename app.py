import os
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from predict import predict_image

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(url_for('index'))

    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    result_path, prediction = predict_image(filepath)

    return render_template('index.html', image_url='/' + result_path, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
