from flask import Flask, request, render_template
import joblib

app = Flask(__name__)
model = joblib.load("D:/model.pkl")

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    input_data = [float(x) for x in request.form.values()]
    prediction = model.predict([input_data])[0]
    return f"Kết quả dự đoán: {'Positive' if prediction == 1 else 'Negative'}"

if __name__ == '__main__':
    app.run(debug=True)