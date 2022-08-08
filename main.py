import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify, render_template

model = load_model('model/model.h5')

application = Flask(__name__)

@application.route('/')
def home():
    return render_template('index.html')

@application.route('/predict', methods=['POST'])
def predict():
    label = {}
    img = Image.open(request.files['img_inp'].stream).convert('L')
    re_img = img.resize((64, 64))
    np_img = np.array(re_img).reshape(1, 64, 64, 1)
    np_img = np_img/np_img.max()

    classify = model.predict(np_img)
    label.update({'Edo': classify[0, 0], 'Hausa': classify[0, 1], 'Igbo': classify[0, 2], 'Yoruba': classify[0, 3]})
    l_class = max(label, key=label.get)
    return render_template('index.html', image_class=l_class)

if __name__ == "__main__":
    application.run(debug=False)
