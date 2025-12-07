from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
import numpy as np

def analyze_sentiment(sentences):
    """
    Analyzes the sentiment of a list of sentences using VADER sentiment analysis.

    Parameters:
    sentences (list of str): A list containing sentences to analyze.

    Returns:
    list of dict: A list of dictionaries containing sentiment scores for each sentence.
    """
    analyzer = SentimentIntensityAnalyzer()
    results = []

    for sentence in sentences:
        result = analyzer.polarity_scores(sentence)
        print(sentence + " " + "\n   " + str(result))
        results.append(result)

    return results

def plot_sentiments(sentences, sentiments):
    """
    Plots the sentiment scores of a list of sentences.

    Parameters:
    sentences (list of str): A list containing sentences.
    sentiments (list of dict): A list of dictionaries containing sentiment scores for each sentence.
    """
    fig, ax = plt.subplots(layout='constrained')

    x = np.arange(len(sentences))
    width = 0.4
    multiplier = 0
    sectionWidth = 2

    neg = []
    neu = []
    pos = []
    compound = []
    for result in sentiments:
        neg.append(result["neg"])
        neu.append(result["neu"])
        pos.append(result["pos"])
        compound.append(result["compound"])
    allScores = {
        'Negative' : neg,
        'Neutral' : neu,
        'Positive' : pos,
        'Compound' : compound
    }

    currentColor = 0
    scoreColors = ['red', 'blue', 'green', 'purple']

    for attribute, score in allScores.items():
        offset = width * multiplier
        rects = ax.bar((x * sectionWidth) + offset, score, width, label=attribute, color=scoreColors[currentColor])
        ax.bar_label(rects, padding=3)
        multiplier += 1
        currentColor += 1

    ax.set_ylabel('Score')
    ax.set_title('Sentiment scores per sentence')
    ax.set_xticks((x * sectionWidth) + (width * 1.5), sentences, rotation=45, ha='right')
    ax.legend(loc='upper left', ncols=4)
    ax.set_ylim(-1.2, 1.5)

    ax.axhline(0, color='black')

    plt.show()

# analyze_sentiment([
#     "I love programming!",
#     "This is the worst movie I've ever seen.",
#     "The weather is okay, not too bad."
# ])

# analyzer = SentimentIntensityAnalyzer()
sentences = ["I love programming!", 
            "This is the worst movie I've ever seen.", 
            "The weather is okay, not too bad."]
# for sentence in sentences:
#     print(analyzer.polarity_scores(sentence))

plot_sentiments(sentences, analyze_sentiment(sentences))