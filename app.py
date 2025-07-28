from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load('calories_model-5.pkl')
scaler = joblib.load('scaler_3.pkl')

@app.route('/')
def home():
    return render_template('index.html', title="TrackMyBurn 🔥")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = [
            float(request.form.get("Gender")),
            float(request.form.get("Age")),
            float(request.form.get("Height")),
            float(request.form.get("Weight")),
            float(request.form.get("Duration")),
            float(request.form.get("Heart_Rate")),
            float(request.form.get("Body_Temp")),
        ]

        print(" Raw Input:", data)

        final_input = scaler.transform([np.array(data)])
        print(" Scaled Input:", final_input)

        prediction = model.predict(final_input)
        print(" Your burnt calories:", prediction)

        return render_template('index.html', prediction_text=f"🔥 Estimated Calories Burnt: {prediction[0]:.2f} kcal")

    except Exception as e:
        return render_template('index.html', prediction_text=f"⚠️ Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
