from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Load the dataset
data = pd.read_csv('data.csv')
X = data['Disease']
y = data['Ayurvedic Remedy']

# Encode text data
tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(X)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_tfidf, y)

@app.route("/", methods=["GET"])
def hello():
    return "Hello, World!"

@app.route("/data",methods=['POST'])
def demo():
    data = request.json
    print("data recived from ardino: "+data)
    return jsonify({"data":data}),400

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict_remedy():
    input_data = request.json
    input_disease = input_data['disease']
    input_disease_tfidf = tfidf_vectorizer.transform([input_disease])
    predicted_remedy = model.predict(input_disease_tfidf)
    return jsonify({'remedy': predicted_remedy[0]})


