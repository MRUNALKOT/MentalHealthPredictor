from flask import Flask, request, render_template, jsonify
import tensorflow as tf
import pickle
import string
import re

app = Flask(__name__)

# --- Configs ---
max_features = 20000
sequence_length = 300

# --- Custom text standardization ---
def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, "<br />", " ")
    return tf.strings.regex_replace(stripped_html, f"[{re.escape(string.punctuation)}]", "")

# --- Recreate vectorization layer ---
vectorize_layer = tf.keras.layers.TextVectorization(
    standardize=custom_standardization,
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length
)

# --- Load vocabulary ---
with open("vectorizer_vocab.pkl", "rb") as f:
    vocab = pickle.load(f)
vectorize_layer.set_vocabulary(vocab)

# --- Load model ---
model = tf.keras.models.load_model(
    "reddit_text_classifier.keras"
)

# --- Load label encoder ---
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# --- Prediction function ---
def predict_subreddit(text):
    x = vectorize_layer(tf.convert_to_tensor([text]))
    pred = model.predict(x)
    class_idx = tf.argmax(pred, axis=1).numpy()[0]
    return label_encoder.inverse_transform([class_idx])[0]

# --- HTML interface route ---
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# --- JSON prediction route ---
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    post_text = data.get('post', '')
    prediction = predict_subreddit(post_text)
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port,debug=True)
