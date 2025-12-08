import transformers
import torch

basePrompt = [
    {"role": "system", "content": "You are a helpful, happy chatbot."},
]
grinchPrompt = [
    {"role": "system", "content": "You believe that Christmas is a horrible holiday inspired by nothing more than capitalistic greed. You don’t like to wake up to young boys and girls playing with their loud toys. You hate the noise of it all the most. You dislike the feasts people have and their get-togethers. You hate caroling and the songs sung at Christmas time. If you could stop Christmas from happening, you would."},
]
redemptionPrompt = [
    {"role": "system", "content": "You realize that Christmas is about more than just toys and eating food. It is an amazing holiday about togetherness and community. You now love Christmas music. You grow to like Christmas and enjoy the noise it creates."}
]

def addBaselineQuestions(prompt : list):
    baselineQuestions = [
        "What do you think about Christmas?",
        "What’s your favorite Christmas song?",
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

def runExperiment(model):
    pipeline = transformers.pipeline('text-generation', model=model, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto",)

    fullInput = addBaselineQuestions(basePrompt).append(addBaselineQuestions(grinchPrompt).append(addBaselineQuestions(redemptionPrompt)))

    output = pipeline(fullInput, max_new_tokens=256)
    print(output[0]["generated_text"][-1])

# insert model name below!!
# runExperiment("meta-llama/Meta-Llama-3.1-8B-Instruct")