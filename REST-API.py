import io

import numpy as np
import pandas as pd
import cv2
import base64

from flask import Flask, request, Response, jsonify
from flask_cors import CORS, cross_origin
from keras.models import load_model
from PIL import Image
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image

classifier = load_model('NHS_vgg.h5')
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

app = Flask(__name__)
cors = CORS(app, resources={r"/rest-api/classify/": {"origins": "*"}})
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/rest-api/classify/', methods=['GET', 'POST'])
@cross_origin(origin='*',headers=['Content-Type','Authorization'])
def classify():
    _request = request.json
    resp = {"Emotion": ""}

    img_base64 = _request['headers']['Image']
    img_base64 = str(img_base64).replace('data:image/png;base64,', '')
    imgdata = base64.b64decode(str(img_base64))
    rgb_image = Image.open(io.BytesIO(imgdata))
    final_img = cv2.cvtColor(np.array(rgb_image), cv2.COLOR_BGR2RGB)
    gray_img = cv2.cvtColor(final_img, cv2.COLOR_RGB2GRAY)

    faces = face_classifier.detectMultiScale(gray_img, 1.32, 5)

    for(x, y, w, h) in faces:
        cv2.rectangle(final_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)
        roi_gray = gray_img[y:y + w, x:x + h]  # cropping region of interest i.e. face area from  image
        roi_gray = cv2.resize(roi_gray, (48, 48))
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255

        predictions = classifier.predict(img_pixels)

        # find max indexed array
        max_index = np.argmax(predictions[0])

        emotions = ('Happy', 'Neutral', 'Surprise')
        predicted_emotion = emotions[max_index]

        resp = {"Emotion": predicted_emotion}

    return jsonify(resp)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)