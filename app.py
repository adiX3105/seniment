from flask import Flask, render_template, request
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download lexicon
nltk.download('vader_lexicon')

app = Flask(__name__)
sia = SentimentIntensityAnalyzer()

@app.route('/', methods=['GET', 'POST'])
def index():
    sentiment = ''
    score = 0
    text = ''

    if request.method == 'POST':
        text = request.form['text']
        scores = sia.polarity_scores(text)
        score = scores['compound']
        if score >= 0.05:
            sentiment = 'Positive'
        elif score <= -0.05:
            sentiment = 'Negative'
        else:
            sentiment = 'Neutral'

    return render_template('index.html', text=text, sentiment=sentiment, score=score)

if __name__ == '__main__':
    app.run(debug=True)
