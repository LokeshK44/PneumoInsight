from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = tf.keras.models.load_model('Y00.h5')

UPLOAD_FOLDER = './images/'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET']) 
def hello():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    if 'imagefile' not in request.files:
        return render_template('index.html', error='No file part')
    
    imagefile = request.files['imagefile']

    if imagefile.filename == '':
        return render_template('index.html', error='No selected file')

    if imagefile and allowed_file(imagefile.filename):
        image_path = os.path.join(UPLOAD_FOLDER, imagefile.filename)
        imagefile.save(image_path)

        img = image.load_img(image_path, target_size=(256, 256))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        prediction = model.predict(img_array)
        predicted_class = "Normal" if np.argmax(prediction) == 0 else "Pneumonia"

        return render_template('index.html', prediction=predicted_class, image_path=image_path, filename=imagefile.filename)

    else:
        return render_template('index.html', error='Invalid file format. Please upload only JPEG images.')

if __name__ == '__main__':
    app.run(port=3000, debug=True)