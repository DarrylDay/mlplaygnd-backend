from flask import Flask, jsonify, request
from flask_cors import CORS 
from mnist.predict import get_prediction
from dotenv import load_dotenv
import os

load_dotenv()

DEBUG = os.getenv("DEBUG") == "True"

app = Flask(__name__)

if DEBUG:
    CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == "POST":
        file = request.files['file']
        img_bytes = file.read()
        prediction = get_prediction(img_bytes)
        return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=DEBUG)