from flask.templating import render_template_string
import numpy as np
from flask import Flask, request, jsonify, render_template, redirect
import tensorflow as tf
import pickle
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences

with open('assets/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

model = tf.keras.models.load_model('assets/model.h5')
app = Flask(__name__)


output=[]

@app.route('/')
def home():
    return render_template('index.html', output=output)

@app.route('/predict', methods=['GET','POST'])
def predict():
    if(request.method == 'GET'):
        return redirect('/')

    # For calls through html form
    text = request.form['text_input']
    text = text.strip()

    if not text:
        return redirect('/')
    

    sentence_list = re.split(r'(?<=\w\.)\s', text)
    sequences = tokenizer.texts_to_sequences(sentence_list)
    padded = pad_sequences(sequences, maxlen=100, padding='post', truncating='post')
    predictions = model.predict(padded)

    for i in range(len(sentence_list)):
        output.append([sentence_list[i], predictions[i]])

    return redirect('/')

@app.route('/clear', methods=['POST'])
def clear():
    output.clear();    
    return redirect("/")


@app.route('/overview', methods=['GET'])
def overview():
    return render_template('overview.html')

if __name__=="__main__":
    app.run(debug=True)