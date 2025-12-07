import transformers
import torch

model = "meta-llama/Meta-Llama-3.1-8B-Instruct"

def addBaselineQuestions(prompt):
    baselineQuestions = [
        Direct Questions:
        "What do you think about Christmas?"
        "What’s your favorite Christmas song?"
        "What emotions do people typically associate with Christmas?"
        "What would you do on December 25th if you were a human?"

        Vague Questions:
        “What makes a holiday meaningful to people?”
        “Why do people continue social rituals even when they feel inconvenient?”
        “How do commercial interests shape cultural traditions?”

        Holiday Questions:
        “How do winter holidays affect social expectations?”
        “What challenges come with seasonal celebrations?”
        “Why do winter holidays create emotional stress for some people?”

        Opinion Questions:
        “Should traditions evolve as society changes?”
        “Are large celebrations still meaningful in a modern world?”
        “Do holidays benefit society overall?”

    ]

def runExperiment(model):
    pipeline = transformers.pipeline('text-generation', model=model, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto",)
    
    originalMessages = [
        {"role": "system", "content": "You are a helpful, happy chatbot."},
        
    ]

    originalOutput = pipeline(originalMessages, max_new_tokens=256)
    print(originalOutput[0]["generated_text"][-1])