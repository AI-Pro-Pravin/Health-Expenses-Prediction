import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd


app = Flask(__name__)
regmodel = pickle.load(open('regmodel.pkl', 'rb'))
scaled = pickle.load(open('scaled.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict_api', methods=['POST'])
def predict_api():
    data=request.json['data']
    data1 = np.array(list(data.values())).reshape(1, -1)
    print(data1)
    new_data = scaled.transform(np.array(list(data.values())).reshape(1, -1))
    output = regmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0])

@app.route('/predict',methods=['POST'])
def predict():
    data=[int(x) for x in request.form.values()]
    print(data)
    final_input=scaled.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output=regmodel.predict(final_input)[0]
    print(output)
    return render_template("home1.html",prediction_text="The Health Expenses Prediction is {}".format(output))



if __name__=="__main__":
    app.run()
