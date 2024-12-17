from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('D:\Study\MachineLearning\Classification_of_College_Student_Pass_Rates\model.pkl', 'rb'))
@app.route('/')
def hello():
    return render_template('index.html')

# prediction function   
@app.route('/predict', methods = ['POST']) 
def predict(): 
    A = [float(x) for x in request.form.values()]
    print(A)
    model_probability = model.predict([A])
    print(A)
    print(model_probability)
    prediction = "Result is %0.2f"%abs(model_probability)
    return render_template('index.html', result = prediction)

if __name__ == "__main__":
    app.run(debug=True)