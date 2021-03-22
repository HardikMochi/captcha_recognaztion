#!/usr/bin/env python
# coding: utf-8




import torchvision.transforms as transforms
import torch
import engine
from model import CaptchaModel
import config
from train import decode_predictions
# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

import numpy as np
from util import base64_to_pil

app = Flask(__name__)

model_path = 'Models/Model'

def preproc_image(image):
    """
    :param image_path: path to the test image
    :return: {'images': image tensor}
    """
    transformer = transforms.Compose([
        transforms.Resize((config.image_height, config.image_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    image = torch.as_tensor(transformer(image), dtype=torch.float)
    image = torch.unsqueeze(image, dim=0)
    image = image.to(config.DEVICE)
    return {'images': image}


def remove_duplicates(x):
    if len(x) < 2:
        return x
    fin = ""
    for j in x:
        if fin == "":
            fin = j
        else:
            if j == fin[-1]:
                continue
            else:
                fin = fin + j
    return fin


def get_predictions(image, model_path):
    le =  np.load('Data/lbl_enc.npy',allow_pickle=True)
    classs = np.load('Data/num_class.npy',allow_pickle=True)
    model = CaptchaModel(num_chars=len(classs))
    model.to(config.DEVICE)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    data = preproc_image(image)

    with torch.no_grad():
        preds, _ = model(**data)

    # Now decode the preds
    preds = decode_predictions(preds, le)
    preds = remove_blanks(preds)
    return preds


# home page
@app.route("/")
def home():
    return render_template("base.html")


@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the image from post request
        img = base64_to_pil(request.json)
        prediction = get_predictions(img, model_path)      

        # Serialize the result, you can add additional fields
        return jsonify(result=prediction)


if __name__ == "__main__":
    app.run(port=5002,debug=False)
 
    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()



