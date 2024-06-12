from dotenv import load_dotenv
load_dotenv()

from flask import Flask, jsonify, request, Response
from flask_cors import CORS
import requests
from backend import supabase, supabase_url
from mnist.predict import get_prediction
import os
import uuid

DEBUG = os.getenv("DEBUG") == "True"
HOST = os.getenv("FLASK_HOST")
PORT = os.getenv("FLASK_PORT")

app = Flask(__name__)
CORS(app)

@app.route('/getBucketImage', methods=['GET'])
def bucket():

    bucket = request.args.get('bucket')
    file = request.args.get('file')
    token = request.args.get('token')

    try:
        user = supabase.auth.get_user(token).user
    except Exception as e:
        print(e)
        return "Invalid authorization token", 400
    except:
        return "Invalid authorization token", 400

    url = f"{supabase_url}/storage/v1/object/authenticated/{bucket}/{user.id}/{file}"

    resp = requests.get(url, headers={"Authorization":f"Bearer {token}"}, stream=True, allow_redirects=False)

    return resp.content, resp.status_code, resp.headers.items()

@app.route('/predict', methods=['POST'])
def predict():

    if request.authorization is None or request.authorization.token is None:
        return "Authorization token not found", 400

    jwt = request.authorization.token

    try:
        user = supabase.auth.get_user(jwt).user
    except Exception as e:
        print(e)
        return "Invalid authorization token", 400
    except:
        return "Invalid authorization token", 400

    if request.method == "POST":
        file = request.files['file']
        img_bytes = file.read()
        prediction = get_prediction(img_bytes)

        bucket = "mnist_images"
        fileName = f"{uuid.uuid4()}.png"
        filePath = f"{user.id}/{fileName}"
        supabase.storage.from_(bucket).upload(filePath, img_bytes, {
            "content-type": "image/png"
        })
        res = supabase.table('mnist_predictions').insert(
            {"user_id": user.id, "prediction": int(prediction), "img_bucket": bucket, "img_filename": fileName}).execute()
        
        return jsonify(res.data[0])
        # return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=DEBUG, host=HOST, port=PORT)