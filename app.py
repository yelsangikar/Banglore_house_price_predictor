from flask import Flask, render_template, request
import pandas as pd
import pickle
import warnings
import numpy as np

warnings.filterwarnings('ignore')

app = Flask(__name__)
data = pd.read_csv('cleaned_data_new.csv')
pipe = pickle.load(open('RidgeModel_new.pkl', 'rb'))


@app.route('/')
def index():
    locations = sorted(data['location'].unique())
    return render_template("index.html", locations=locations)


@app.route('/predict', methods=['POST'])
def predict():
    location = request.form.get('location')
    bhk = request.form.get('bhk')
    bath = request.form.get('bath')
    sqft = request.form.get('sqft')

    print(location, bhk, bath, sqft)
    #inputs = pd.DataFrame([['location', 'sqft', 'bath', 'bhk']], columns= X_train.columns)
    #input = pd.DataFrame([['location', 'sqft', 'bath', 'bhk']], columns=['location', 'total_sqft', 'bath', 'BHK'])
    input = pd.DataFrame([[location, sqft, bath, bhk]], columns=['location', 'total_sqft', 'bath', 'BHK'])

    prediction = pipe.predict(input)[0] * 1e5
    return str(np.round(prediction,2))


if __name__ == '__main__':
    app.run(debug=True)
