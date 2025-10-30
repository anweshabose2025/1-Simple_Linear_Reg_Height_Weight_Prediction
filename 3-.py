# (D:\Udemy\Complete_DSMLDLNLP_Bootcamp\UPractice1\venv) D:\Udemy\Complete_DSMLDLNLP_Bootcamp\UPractice2\1-Simple_Linear_Reg>python 3-.py

from flask import Flask, request, render_template
import pickle
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

with open('model.pkl', 'rb') as file:
    linear_model = pickle.load(file)
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

@app.route('/', methods = ['GET', 'POST'])
def welcome():
    if request.method == 'POST':
        weight = float(request.form['weight'])
        new_weight_df = pd.DataFrame([weight], columns=['Weight'])
        new_weight_df = scaler.transform(new_weight_df)
        predicted_height = linear_model.predict(new_weight_df)
        return f"<h1>Predict height is: {predicted_height[0][0]:.2f}<h1>"
    return render_template('front_page.html')

if __name__ == "__main__":
    app.run(debug = True)
