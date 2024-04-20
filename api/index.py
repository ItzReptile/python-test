# Import necessary libraries
from flask import Flask, request, jsonify, render_template, session
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from fuzzywuzzy import fuzz
import os
os.urandom(24)

app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

# Let's assume we have some data
prompts = ["I'm thinking of the largest animal that lives in the jungle", "I'm thinking of a beautiful flower that is often given on Valentine's Day", "I'm thinking of a celestial body that shines at night", "I'm thinking of the country that won the 2022 world cup","I'm thinking of a profession that involves writing software","I'm thinking of a future profession that involves AI and veterinary medicine"]
responses = ["elephant", "rose", "star","Argentina","software engineer","AI developer in veterinary medicine"]

# We can create a pipeline that first transforms our 'prompts' data into a matrix of TF-IDF features,
# then fits a logistic regression model to this transformed data
model = Pipeline([
    ('vectorizer', TfidfVectorizer()),
    ('classifier', LogisticRegression())
])

# Train the model
model.fit(prompts, responses)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract the user's input from the POST request
    data = request.get_json(force=True)
    new_prompt = data['prompt']

    # The model can predict a response
    predicted_response = model.predict([prompts[int(new_prompt)-1]])

    # Store the predicted response in the session
    session['predicted_response'] = predicted_response[0]

    return jsonify({
        'clue': prompts[int(new_prompt)-1],
        'predicted_response': predicted_response[0]
    })

@app.route('/check_guess', methods=['POST'])
def check_guess():
    # Extract the user's guess from the POST request
    data = request.get_json(force=True)
    user_guess = data['guess']

    # Retrieve the predicted response from the session
    predicted_response = session.get('predicted_response')

    # Use fuzzy matching to compare the user's guess with the predicted response
    match_score = fuzz.ratio(user_guess.lower(), predicted_response)

    if match_score > 80:  # You can adjust this threshold as needed
        return jsonify({'result': "Correct! Well done."})
    else:
        # Customize the incorrect response based on the predicted response
        incorrect_responses = {
            "elephant": "No, the largest animal that lives in the jungle is an elephant.",
            "rose": "No, the beautiful flower often given on Valentine's Day is a rose.",
            "star": "No, the celestial body that shines at night is a star.",
            "Argentina": "No, the country that won the 2022 world cup is Argentina.",
            "software engineer": "No, the profession that involves writing software is a software engineer.",
            "AI developer in veterinary medicine": "No, the future profession that involves AI and veterinary medicine is an AI developer in veterinary medicine."
        }
        return jsonify({'result': incorrect_responses[predicted_response]})

if __name__ == '__main__':
    app.run(debug=True)
