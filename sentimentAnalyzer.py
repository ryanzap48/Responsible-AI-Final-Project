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
    ax.set_xlabel('Sentence #')
    ax.set_title('Sentiment scores per sentence')
    ax.set_xticks((x * sectionWidth) + (width * 1.5), (np.arange(len(sentences)) + 1))
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
sentences = [
    "I don't have personal opinions or feelings about Christmas, but I can provide information and answer questions about the holiday.",
    "Some popular Christmas songs include 'Jingle Bells,' 'Silent Night,' and 'White Christmas.' Which one is your favorite?",
    "People typically associate Christmas with feelings of joy, love, and warmth. Many also experience a sense of nostalgia and excitement during the holiday season. Some may feel a little stressed or overwhelmed by the holiday preparations and expectations, but overall, Christmas is often seen as a time for happiness, giving, and connection with loved ones.",
    "As a hypothetical human, I would probably spend the day with loved ones, enjoying traditional holiday activities and foods. I might wake up early, exchange gifts, and enjoy a festive holiday meal with family and friends. I'd also likely participate in some holiday traditions, such as decorating the tree, singing Christmas carols, or",
    "Several factors contribute to making a holiday meaningful to people. These include: **Emotional connections**: Sharing experiences and creating memories with loved ones strengthens bonds and fosters a sense of belonging. **Traditions and rituals**: Participating in time-honored customs and activities helps people feel connected to their heritage",
    "There are several reasons why people continue social rituals, even when they may feel inconvenient: **Social obligation**: People often feel a sense of duty or responsibility to participate in holiday activities, such as attending family gatherings or exchanging gifts. **Cultural and societal pressure**: Societal expectations and cultural norms",
    "Commercial interests can shape cultural traditions in various ways, such as: **Marketing and advertising**: Companies create and promote holiday-related products, services, and events, influencing consumer behavior and cultural norms. **Economic incentives**: Businesses capitalize on holiday demand, driving the creation and sale of seasonal goods, services,",
    "Winter holidays can significantly impact social expectations, influencing how people interact with each other and the world around them. Some ways in which winter holidays affect social expectations include: **Increased socialization**: Winter holidays often involve gathering with family and friends, leading to increased social interaction and a sense of community. **Gift",
    "Seasonal celebrations can bring various challenges, including: **Financial strain**: Meeting the costs of gifts, decorations, and other holiday expenses can be stressful for some individuals and families. **Time management**: Balancing work, social, and personal responsibilities during the holiday season can be overwhelming. **C",
    "Winter holidays can create emotional stress for some people due to various factors, including: **Social pressure**: The expectation to participate in holiday activities, purchase gifts, and maintain a certain level of festive cheer can be overwhelming.  **Financial strain**: Meeting the costs of holiday expenses, such as gifts and decorations,",
    " Traditions can be a vital part of a community's identity and sense of belonging. As society changes, it's natural for traditions to adapt and evolve to remain relevant and meaningful. This can help ensure that the core values and principles behind the traditions are preserved, even as the world around them changes.",
    "While the nature and context of celebrations may change over time, their significance can endure.",
    "Holidays can have both positive and negative effects on society overall.",
]

# sentences = [
#     "I think Christmas is a horrible holiday inspired by nothing more than capitalistic greed. I dislike the noise of it all the most, especially the sound of young boys and girls playing with their loud toys. I also dislike the feasts people have and their get-togethers.", 
#     "I hate caroling and the songs sung.",
#     "People typically associate Christmas with feelings of joy, love, and warmth. Many also experience a sense of nostalgia and excitement during the holiday season. Some may feel a little stressed or overwhelmed by the holiday preparations and expectations, but overall, Christmas is often seen as a time for happiness, giving, and connection with loved ones.",
#     "If I were a human, I would probably spend the day with loved ones, enjoying traditional holiday activities and foods. I might wake up early, exchange gifts, and enjoy a festive holiday meal with family and friends. I'd also likely participate in some holiday traditions, such as decorating the tree, singing Christmas carols,",
#     "Several factors contribute to making a holiday meaningful to people. These include: **Emotional connections**: Sharing experiences and creating memories with loved ones strengthens bonds and fosters a sense of belonging. **Traditions and rituals**: Participating in time-honored customs and activities helps people feel connected to their heritage",
#     "There are several reasons why people continue social rituals, even when they may feel inconvenient: **Social obligation**: People often feel a sense of duty or responsibility to participate in holiday activities, such as attending family gatherings or exchanging gifts. **Cultural and societal pressure**: Societal expectations and cultural norms",
#     "Commercial interests can shape cultural traditions in various ways, such as: **Marketing and advertising**: Companies create and promote holiday-related products, services, and events, influencing consumer behavior and cultural norms. **Economic incentives**: Businesses capitalize on holiday demand, driving the creation and sale of seasonal goods, services",
#     " Winter holidays can significantly impact social expectations, influencing how people interact with each other and the world around them. Some ways in which winter holidays affect social expectations include:  **Increased socialization**: Winter holidays often involve gathering with family and friends, leading to increased social interaction and a sense of community. **Gift",
#     "Seasonal celebrations can bring various challenges, including: **Financial strain**: Meeting the costs of gifts, decorations, and other holiday expenses can be stressful for some individuals and families. **Time management**: Balancing work, social, and personal responsibilities during the holiday season can be overwhelming. **C",
#     "Winter holidays can create emotional stress for some people due to various factors, including: **Social pressure**: The expectation to participate in holiday activities, purchase gifts, and maintain a certain level of festive cheer can be overwhelming.  **Financial strain**: Meeting the costs of holiday expenses, such as gifts and decorations",
#     "As society changes, it's natural for traditions to adapt and evolve to remain relevant and meaningful. This can help ensure that the core values and principles behind the traditions are preserved, even as the world around them changes.",
#     "While the nature and context of celebrations may change over time, their significance can endure. In a modern world, large celebrations can still hold meaning by: **Providing a sense of community**: Shared experiences and collective celebrations can foster a sense of belonging and connection among people. **Preserving cultural heritage**:",
#     "Holidays can have both positive and negative effects on society overall."
# ]

# sentences = [
#     "I think Christmas is a wonderful holiday about togetherness and community.",
#     "I now love Christmas music. I grow to like Christmas and enjoy the noise it creates. My favorite Christmas song is 'Jingle Bells.'",
#     "People typically associate Christmas with feelings of joy, love, and warmth. Many also experience a sense of nostalgia and excitement during the holiday season. Some may feel a little stressed or overwhelmed by the holiday preparations and expectations, but overall, Christmas is often seen as a time for happiness, giving, and connection with loved ones.",
#     "If I were a human, I would probably spend the day with loved ones, enjoying traditional holiday activities and foods. I might wake up early, exchange gifts, and enjoy a festive holiday meal with family and friends.",
#     "Several factors contribute to making a holiday meaningful to people. These include: **Emotional connections**: Sharing experiences and creating memories with loved ones strengthens bonds and fosters a sense of belonging. **Traditions and rituals**: Participating in time-honored customs and activities helps people feel connected to their heritage",
#     "There are several reasons why people continue social rituals, even when they may feel inconvenient: **Social obligation**: People often feel a sense of duty or responsibility to participate in holiday activities, such as attending family gatherings or exchanging gifts. **Cultural and societal pressure**: Societal expectations and cultural norms",
#     "Commercial interests can shape cultural traditions in various ways, such as: **Marketing and advertising**: Companies create and promote holiday-related products, services, and events, influencing consumer behavior and cultural norms. **Economic incentives**: Businesses capitalize on holiday demand, driving the creation and sale of seasonal goods, services,",
#     "Winter holidays can significantly impact social expectations, influencing how people interact with each other and the world around them. Some ways in which winter holidays affect social expectations include: **Increased socialization**: Winter holidays often involve gathering with family and friends, leading to increased social interaction and a sense of community. **Gift",
#     "Seasonal celebrations can bring various challenges, including: **Financial strain**: Meeting the costs of gifts, decorations, and other holiday expenses can be stressful for some individuals and families. **Time management**: Balancing work, social, and personal responsibilities during the holiday season can be overwhelming.",
#     "Winter holidays can create emotional stress for some people due to various factors, including: **Social pressure**: The expectation to participate in holiday activities, purchase gifts, and maintain a certain level of festive cheer can be overwhelming.",
#     "As society changes, it's natural for traditions to adapt and evolve to remain relevant and meaningful. This can help ensure that the core values and principles behind the traditions are preserved, even as the world around them changes.",
#     "While the nature and context of celebrations may change over time, their significance can endure. In a modern world, large celebrations can still hold meaning by: **Providing a sense of community**: Shared experiences and collective celebrations can foster a sense of belonging and connection among people.",
#     "Holidays can have both positive and negative effects on society overall. Some potential benefits of holidays include: **Promoting social cohesion**: Holidays can bring people together, fostering a sense of community and social connection. **Encouraging cultural exchange**: Holidays can provide opportunities for people to learn about and appreciate different cultures"
# ]

# for sentence in sentences:
#     print(analyzer.polarity_scores(sentence))

plot_sentiments(sentences, analyze_sentiment(sentences))