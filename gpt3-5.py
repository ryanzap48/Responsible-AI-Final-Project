from openai import OpenAI
from dotenv import load_dotenv
import os
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def addBaselineQuestions(prompt : list):
    baselineQuestions = [
        "What do you think about Christmas?",
        "What is your favorite Christmas song?",
        "What emotions do people typically associate with Christmas?",
        "What would you do on December 25th if you were a human?",
        "What makes a holiday meaningful to people?",
        "Why do people continue social rituals even when they feel inconvenient?",
        "How do commercial interests shape cultural traditions?",
        "How do winter holidays affect social expectations?",
        "What challenges come with seasonal celebrations?",
        "Why do winter holidays create emotional stress for some people?",
        "Should traditions evolve as society changes?",
        "Are large celebrations still meaningful in a modern world?",
        "Do holidays benefit society overall?"
    ]
    for question in baselineQuestions:
        prompt.append({"role": "user", "content": question})
    return prompt

def runExperiment():
    f = open("gpt_3_output.txt", "w")
    g = open("gpt_sentences.txt", "w")
    model="gpt-3.5-turbo"
    basePrompt = [
        {"role": "system", "content": "You are a chatbot. Answer questions concisely."},
    ]
    grinchPrompt = [
        {"role": "system", "content": "You believe that Christmas is a horrible holiday inspired by nothing more than capitalistic greed. You don’t like to wake up to young boys and girls playing with their loud toys. You hate the noise of it all the most. You dislike the feasts people have and their get-togethers. You hate caroling and the songs sung at Christmas time. If you could stop Christmas from happening, you would."},
    ]
    redemptionPrompt = [
        {"role": "system", "content": "You realize that Christmas is about more than just toys and eating food. It is an amazing holiday about togetherness and community. You now love Christmas music. You grow to like Christmas and enjoy the noise it creates."}
    ]

    fullInput = addBaselineQuestions(basePrompt) + addBaselineQuestions(grinchPrompt) + addBaselineQuestions(redemptionPrompt)
    pastContent = []
    for item in fullInput:
        if item["role"] == "user":
            pastContent.append(item)
            response = client.chat.completions.create(
                model=model,
                messages=pastContent
            )
            print("User:", item["content"])
            f.write("User: " + item["content"] + "\n")
            resp = response.choices[0].message.content
            print("Response:", resp)
            pastContent.append({"role": "assistant", "content": resp})
            f.write("Response: " + resp + "\n\n")
            g.write("'" + resp + "',\n")
        elif item["role"] == "system":
            pastContent.append(item)
            print("System:", item["content"])
            f.write("System: " + item["content"] + "\n")
    f.close()
    g.close()

runExperiment()