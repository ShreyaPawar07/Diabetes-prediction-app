from flask import Flask , request,jsonify,render_template
import pickle
import numpy as np
from sklearn import svm


model = pickle.load(open('model1.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

# as we are getting data from form we use request.form
# post method is used to take values only not urls

@app.route('/predict',methods=['POST'])
def predict():
    # Pregnancies = request.form.get('Pregnancies')
    Glucose = request.form.get('Glucose')
    # BloodPressure = request.form.get('BloodPressure')
    SkinThickness  = request.form.get('SkinThickness')
    Insulin    = request.form.get('Insulin')
    # BMI = request.form.get('BMI')
    # DiabetesPedigreeFunction = float(request.form.get('DiabetesPedigreeFunction'))
    Age = request.form.get('Age')
    
    input_query = np.array([[Age,Insulin,SkinThickness,Glucose]])
    # input array for storing values entered by user 
    result = model.predict(input_query)[0]
    if result[0] == 1:
        result = 'diabetic'
    else:
        result = 'not diabetic'
    # predicting with help of model

    return render_template('index.html',result=result)

if __name__ == '__main__' :
    app.run(debug=True)
   
   