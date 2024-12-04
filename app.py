from flask import Flask, request, jsonify, render_template
import PyPDF2
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pickle
import os

import pandas as pd

app = Flask(__name__)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
with open('irsi_data.pkl', 'rb') as file:
    model_flower = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler_flower = pickle.load(file)


def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    lemmatizer = WordNetLemmatizer()
    return " ".join(
        lemmatizer.lemmatize(word)
        for word in tokens
        if word.isalnum() and word not in stop_words
    ) 
def resume_similarity(job_description, resume_text):
    results = []  

    
    job_description = preprocess_text(job_description)
    resume_processed = [preprocess_text(resume_txt) for resume_txt in resume_text]
    
    
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf = vectorizer.fit_transform([job_description] + resume_processed)
    
    
    similarity_scores = cosine_similarity(tfidf[0:1], tfidf[1:])
    
  
    top_index = similarity_scores.argsort()[0][-5:][::-1]
    top_res = [resume_text[i] for i in top_index]

    
    for idx, resume in zip(top_index, top_res):
        similarity_percentage = round(similarity_scores[0][idx] * 100, 2)
        results.append({
            'resume_index': idx + 1,
            'similarity_score': similarity_percentage,
            'resume_text': resume[:500]  
        })
    return results
      
    
def pdf(path):
    with open(path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/flowers', methods=['POST', 'GET'])
def predict():
    
    sepal_length = float(request.form.get('sepal_length'))
    sepal_width = float(request.form.get('sepal_width'))
    petal_length = float(request.form.get('petal_length'))
    petal_width = float(request.form.get('petal_width'))

    
    input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]], 
                              columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])

    scaled_data = scaler_flower.transform(input_data)

   
    prediction = model_flower.predict(scaled_data)
    prediction_text = f"Flower Predicted: {prediction[0]}"

    return render_template('index.html', prediction_text=prediction_text)


# 
@app.route('/resume', methods=['POST', 'GET'])
def resume():
    if request.method == 'POST':  
        
        job_description = request.form['job_description'] 
        resume_files = request.files.getlist('resumes')  
        
        resume_texts = []  
        for resume_file in resume_files:  
            resume_path = os.path.join('static', resume_file.filename) 
            resume_file.save(resume_path) 
            resume_text = pdf(resume_path) 
            resume_texts.append(resume_text) 
        
        results = resume_similarity(job_description, resume_texts) 
        return render_template('resume.html', results=results) 
    
    return render_template('resume.html', results=None) 


if __name__ == '__main__':
    app.run(debug=True)
