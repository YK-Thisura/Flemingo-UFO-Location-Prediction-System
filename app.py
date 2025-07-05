from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
import re
import os
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))

# US + Canada states/provinces
STATE_FULL_NAMES = {
    'al': 'Alabama', 'ak': 'Alaska', 'az': 'Arizona', 'ar': 'Arkansas', 'ca': 'California',
    'co': 'Colorado', 'ct': 'Connecticut', 'de': 'Delaware', 'fl': 'Florida', 'ga': 'Georgia',
    'hi': 'Hawaii', 'id': 'Idaho', 'il': 'Illinois', 'in': 'Indiana', 'ia': 'Iowa', 'ks': 'Kansas',
    'ky': 'Kentucky', 'la': 'Louisiana', 'me': 'Maine', 'md': 'Maryland', 'ma': 'Massachusetts',
    'mi': 'Michigan', 'mn': 'Minnesota', 'ms': 'Mississippi', 'mo': 'Missouri', 'mt': 'Montana',
    'ne': 'Nebraska', 'nv': 'Nevada', 'nh': 'New Hampshire', 'nj': 'New Jersey', 'nm': 'New Mexico',
    'ny': 'New York', 'nc': 'North Carolina', 'nd': 'North Dakota', 'oh': 'Ohio', 'ok': 'Oklahoma',
    'or': 'Oregon', 'pa': 'Pennsylvania', 'ri': 'Rhode Island', 'sc': 'South Carolina',
    'sd': 'South Dakota', 'tn': 'Tennessee', 'tx': 'Texas', 'ut': 'Utah', 'vt': 'Vermont',
    'va': 'Virginia', 'wa': 'Washington', 'wv': 'West Virginia', 'wi': 'Wisconsin', 'wy': 'Wyoming',
    'dc': 'District of Columbia', 'ab': 'Alberta', 'bc': 'British Columbia', 'mb': 'Manitoba',
    'nb': 'New Brunswick', 'nl': 'Newfoundland and Labrador', 'ns': 'Nova Scotia',
    'nt': 'Northwest Territories', 'nu': 'Nunavut', 'on': 'Ontario', 'pe': 'Prince Edward Island',
    'qc': 'Quebec', 'sk': 'Saskatchewan', 'yt': 'Yukon', 'nf': 'Newfoundland and Labrador', 'pq': 'Quebec'
}

class UFOPredictor:
    def __init__(self, model_path="model/ufo_sighting_model.joblib"):
        self.model = None
        self.tfidf = None
        self.label_encoder = None
        self.model_path = model_path
        self._load_model()

    def _load_model(self):
        if os.path.exists(self.model_path):
            payload = joblib.load(self.model_path)
            self.model = payload['model']
            self.tfidf = payload['tfidf']
            self.label_encoder = payload['label_encoder']
        else:
            raise FileNotFoundError("Model file not found.")

    def _clean_text(self, text):
        text = re.sub(r'[^a-zA-Z\s]', '', text, re.I | re.A).lower()
        tokens = text.split()
        tokens = [word for word in tokens if word not in STOPWORDS]
        return ' '.join(tokens)

    def predict(self, description, sighting_time, duration_sec):
        dt = datetime.strptime(sighting_time, "%Y-%m-%d %H:%M")
        duration_sec = float(duration_sec)

        year, month, hour, weekday = dt.year, dt.month, dt.hour, dt.weekday()
        clean_text = self._clean_text(description)
        text_vec = self.tfidf.transform([clean_text]).toarray()
        time_features = np.array([year, month, hour, weekday, duration_sec]).reshape(1, -1)
        input_vec = np.concatenate([time_features, text_vec], axis=1)

        pred_label = self.model.predict(input_vec)
        pred_abbr = self.label_encoder.inverse_transform(pred_label)[0]
        return STATE_FULL_NAMES.get(pred_abbr.lower(), pred_abbr.upper())

# Flask App
app = Flask(__name__)
predictor = UFOPredictor()

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    error = None

    if request.method == 'POST':
        try:
            desc = request.form['description']
            time = request.form['sighting_time']
            duration = request.form['duration']
            prediction = predictor.predict(desc, time, duration)
        except Exception as e:
            error = f"Error: {str(e)}"

    return render_template('index.html', prediction=prediction, error=error)

if __name__ == "__main__":
    app.run(debug=True)

