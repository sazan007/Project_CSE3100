import numpy as np
import pandas as pd
import cv2
from flask import Flask,request,render_template
import keras
import pickle

app = Flask(__name__)

model = pickle.load(open('model.pkl','rb'))

model2 = pickle.load(open('model2.pkl','rb'))

model3 =keras.models.load_model('model3.h5')

@app.route("/") 
def home():
    return render_template("index.html")

@app.route("/consultancy") 
def consult():
   return render_template("consultancy.html")

@app.route("/breastcancer")
def func():
    return render_template("breast_cancer.html")

@app.route("/parkinsons")
def parkinsons():
    return render_template("pd.html")

@app.route("/bch")
def bchisto():
    return render_template("histo_breast.html")

@app.route("/about")
def about():
    return render_template("About.html")


@app.route("/predict1",methods=['POST'])
def predict1():
    input_features = [float(x) for x in request.form.values()]
    features_value = [np.array(input_features)]

    inp_reshape  = np.reshape(features_value,(1,-1))
    

    prediction = model.predict(inp_reshape)

    if(prediction[0]==1):
        res = "no Breast Cancer"
    else:
        res = "Breast Cancer"
    

    
    return render_template("breast_cancer.html",prediction_text='Patient has {}'.format(res))



@app.route("/predict2",methods=['POST'])
def predict2():
    input_features = [float(x) for x in request.form.values()]
    features_value = [np.array(input_features)]

    inp_reshape  = np.reshape(features_value,(1,-1))

    
    prediction = model2.predict(inp_reshape)

    if(prediction[0]==1):
        res = "PARKINSONS DISEASE!"
    else:
        res = "NO PARKINSONS DISEASE!"
    

    
    return render_template("pd.html",prediction_text2='PATIENT HAS {}'.format(res))

@app.route('/predict3',methods=['POST'])
def predict3():
    res='OK'
    ImageFile=request.files['ImageFile']
    image_path="./static/image/" + ImageFile.filename
    ImageFile.save(image_path)

    img_array = cv2.imread(image_path) 
    filee=img_array
    filee = np.reshape(filee, (-1, filee.shape[0], filee.shape[1],filee.shape[2]))
    filee = filee/255.0
    ans=model3.predict(filee)
    output = np.argmax(ans[0])
    if output==0:
     res='NO BREAST CANCER'
    else:
     res='BREAST CANCER'

    return render_template("histo_breast.html",prediction_text3='PATIENT HAS {}'.format(res),image_path=image_path)

   

if __name__ == "__main__":
    app.run(debug=True)
