import sys
import openai
import time

openai.api_key = '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'

def complete(messages, conversational_LLM):
    if conversational_LLM == 'chatgpt':
        return complete_chatgpt(messages)
    else:
        print(f'>>Using Vicuna Model')
        raise NotImplementedError
        # return complete_vicuna(prompt, model, tokenizer)

def complete_chatgpt(messages):
    received = False
    temperature = 0
    while not received:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=temperature,
                frequency_penalty=0.2,
                n=None)
            raw_generated_text = response["choices"][0]["message"]['content']   
            received=True
        except:
            error = sys.exc_info()[0]
            if error == openai.error.InvalidRequestError: # something is wrong: e.g. prompt too long
                print(f"InvalidRequestError\nPrompt error.\n\n")
                print("prompt too long")
                return "prompt too long"
            if error == AssertionError:
                print("Assert error:", sys.exc_info()[1])
                # assert False
            else:
                print("API error:", error)
            time.sleep(1)
    return raw_generated_text#, messages