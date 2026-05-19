from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load model and vectorizer
model = pickle.load(open('fake_review_model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    review = request.form['review']

    data = vectorizer.transform([review])

    prediction = model.predict(data)

    if prediction[0] == 1:
        result = "Fake Review"
    else:
        result = "Genuine Review"

    return render_template('index.html', prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)