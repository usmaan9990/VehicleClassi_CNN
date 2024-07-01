from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Dictionary for class labels
dic = {0:'SUV', 1:'BUS', 2:'Family Sedan', 3:'Fire Engine', 4:'Heavy truck', 5:'Jeep', 6:'Mini Bus', 
       7:'Racing Car', 8:'Taxi', 9:'Truck'}

# Load the model
model = load_model('CNN_model.h5')
model.make_predict_function()

def predict_label(img_path):
    i = image.load_img(img_path, target_size=(150,150))
    i = image.img_to_array(i)/255.0
    i = np.expand_dims(i, axis=0)
    pred = model.predict(i)
    p = np.argmax(pred, axis=1)
    return dic[p[0]]

# Routes
@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("index.html")

@app.route("/about")
def about_page():
    return "Please subscribe to Artificial Intelligence Hub..!!!"

@app.route("/submit", methods=['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        try:
            img = request.files['my_image']
            img_path = "static/" + img.filename	
            img.save(img_path)
            p = predict_label(img_path)
            return render_template("index.html", prediction=p, img_path=img_path)
        except Exception as e:
            return str(e)

if __name__ =='__main__':
    app.run(debug=True)
