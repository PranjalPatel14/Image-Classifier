import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model #type: ignore
import numpy as np 
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

model = load_model(r'D:\Projects\Image Classifier\Image_classify.keras')
data_cat = ['apple',
 'banana',
 'beetroot',
 'bell pepper',
 'cabbage',
 'capsicum',
 'carrot',
 'cauliflower',
 'chilli pepper',
 'corn',
 'cucumber',
 'eggplant',
 'garlic',
 'ginger',
 'grapes',
 'jalepeno',
 'kiwi',
 'lemon',
 'lettuce',
 'mango',
 'onion',
 'orange',
 'paprika',
 'pear',
 'peas',
 'pineapple',
 'pomegranate',
 'potato',
 'raddish',
 'soy beans',
 'spinach',
 'sweetcorn',
 'sweetpotato',
 'tomato',
 'turnip',
 'watermelon']
img_height = 180
img_width = 180

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            image_load = tf.keras.utils.load_img(filepath, target_size=(img_height,img_width))
            img_arr = tf.keras.utils.array_to_img(image_load)
            img_bat=tf.expand_dims(img_arr,0)

            predict = model.predict(img_bat)
            score = tf.nn.softmax(predict)
            prediction = data_cat[np.argmax(score)]
            accuracy = str(np.max(score)*100)
            
            return render_template('index.html', filename=file.filename ,predicted_class=prediction, accuracy=accuracy, image=filepath)
    return render_template('index.html')
if __name__ == "__main__":
    app.run(debug=False)

        


