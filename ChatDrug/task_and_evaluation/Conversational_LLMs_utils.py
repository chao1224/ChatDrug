import sys
import openai
import time
import torch
import sys

openai.api_key = YOUR_API_KEY

def complete(messages, model, tokenizer, conversational_LLM, drug_type, round_index=None):
    if conversational_LLM == 'chatgpt':
        return complete_chatgpt(messages)
    elif conversational_LLM == 'llama2':
        return complete_llama(messages, model, tokenizer)
    elif conversational_LLM == 'galactica':
        if drug_type=="molecule":
            return complete_galactica_molecule(messages, model, tokenizer, round_index)
        elif drug_type=="peptide":
            return complete_galactica_peptide(messages, model, tokenizer, round_index)
        elif drug_type == 'protein':
            return complete_galactica_protein(messages, model, tokenizer, round_index)
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


def complete_galactica_molecule(
    messages,
    model,
    tokenizer,
    round_index,
):
    with torch.no_grad():
        if round_index==0:
            input_text = messages[1]['content']
            input_text = input_text+" [START_I_SMILES]"
        else:
            input_text = ""
            for i in range(len(messages)-1):
                if i%2==0:
                    input_text+= messages[i+1]['content']+" [START_I_SMILES]"
                if i%2==1:
                    input_text+= messages[i+1]['content']+"[END_I_SMILES]"+"\n\n"
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to('cuda')
        outputs = model.generate(
            input_ids,
            max_new_tokens=100,
            do_sample=True,
            top_p=0.95,
            temperature=1.0,
            use_cache=True,
            top_k=50,
            repetition_penalty=1.0,
            length_penalty=1,
        )

        output_text = tokenizer.decode(outputs[0])
        output_text_list = output_text.split("[START_I_SMILES]")
        output_text = output_text_list[2+round_index*3].strip()
        output_text_list = output_text.split("[END_I_SMILES]")
        output_text = output_text_list[0].strip()

    return output_text


def complete_galactica_peptide(
    messages,
    model,
    tokenizer,
    round_index,
):
    with torch.no_grad():
        if round_index==0:
            input_text = messages[1]['content']
        else:
            print(messages)
            input_text = ""
            for i in range(len(messages)-1):
                if i%2==0:
                    input_text+= messages[i+1]['content']
                if i%2==1:
                    input_text+= messages[i+1]['content']+"\n\n"
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to('cuda')
        outputs = model.generate(
            input_ids,
            max_new_tokens=100,
            do_sample=True,
            top_p=0.95,
            temperature=1.0,
            use_cache=True,
            top_k=50,
            repetition_penalty=1.0,
            length_penalty=1,
        )

        output_text = tokenizer.decode(outputs[0])
        output_text_list = output_text.split("Answer:")
        output_text = output_text_list[1+round_index].strip()
        output_text_list = output_text.split("Question:")
        output_text = output_text_list[0].strip()

    return output_text


def complete_galactica_protein(
    messages,
    model,
    tokenizer,
    round_index,
):
    with torch.no_grad():
        if round_index==0:
            input_text = messages[1]['content']
            input_text = input_text+" [START_AMINO]"
        else:
            input_text = ""
            for i in range(len(messages)-1):
                if i%2==0:
                    input_text+= messages[i+1]['content']+" [START_AMINO]"
                if i%2==1:
                    input_text+= messages[i+1]['content']+"[END_AMINO]"+"\n\n"
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to('cuda')
        outputs = model.generate(
            input_ids,
            max_new_tokens=100,
            do_sample=True,
            top_p=0.95,
            temperature=1.0,
            use_cache=True,
            top_k=50,
            repetition_penalty=1.0,
            length_penalty=1,
        )

        output_text = tokenizer.decode(outputs[0])
        output_text_list = output_text.split("[START_AMINO]")
        output_text = output_text_list[2+round_index*3].strip()
        output_text_list = output_text.split("[END_AMINO]")
        output_text = output_text_list[0].strip()

    return output_text


B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

def format_tokens(dialogs, tokenizer):
    prompt_tokens = []
    for dialog in dialogs:
        if dialog[0]["role"] != "system":
                dialog = [
                    {
                        "role": "system",
                        "content": DEFAULT_SYSTEM_PROMPT,
                    }
                ] + dialog
        dialog = [
            {
                "role": dialog[1]["role"],
                "content": B_SYS
                + dialog[0]["content"]
                + E_SYS
                + dialog[1]["content"],
            }
        ] + dialog[2:]
        assert all([msg["role"] == "user" for msg in dialog[::2]]) and all(
            [msg["role"] == "assistant" for msg in dialog[1::2]]
        ), (
            "model only supports 'system','user' and 'assistant' roles, "
            "starting with user and alternating (u/a/u/a/u...)"
        )
        """
        Please verify that yout tokenizer support adding "[INST]", "[/INST]" to your inputs.
        Here, we are adding it manually.
        """

        dialog_tokens = sum(
            [
                tokenizer.encode(
                    f"{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} ",
                )
                for prompt, answer in zip(dialog[::2], dialog[1::2])
            ],
            [],
        )
        assert (
            dialog[-1]["role"] == "user"
        ), f"Last message must be from user, got {dialog[-1]['role']}"
        dialog_tokens += tokenizer.encode(
            f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST}",
        )
        prompt_tokens.append(dialog_tokens)
    return prompt_tokens


def complete_llama(
    dialogs,
    model,
    tokenizer,
    max_new_tokens =1024, #The maximum numbers of tokens to generate
    seed: int=42, #seed value for reproducibility
    do_sample: bool=True, #Whether or not to use sampling ; use greedy decoding otherwise.
    use_cache: bool=True,  #[optional] Whether or not the model should use the past last key/values attentions Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.
    top_p: float=0.95, # [optional] If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
    temperature: float=1.0, # [optional] The value used to modulate the next token probabilities.
    top_k: int=50, # [optional] The number of highest probability vocabulary tokens to keep for top-k-filtering.
    repetition_penalty: float=1.0, #The parameter for repetition penalty. 1.0 means no penalty.
    length_penalty: int=1, #[optional] Exponential penalty to the length that is used with beam-based generation.
    **kwargs
):
    # Set the seeds for reproducibility
    torch.cuda.manual_seed(seed)
    chats = format_tokens([dialogs], tokenizer)
    chat = chats[0]

    with torch.no_grad():
        tokens= torch.tensor(chat).long()
        tokens= tokens.unsqueeze(0)
        tokens= tokens.to("cuda:0")
        outputs = model.generate(
            tokens,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_p=top_p,
            temperature=temperature,
            use_cache=use_cache,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            **kwargs
        )

        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        output_text_list = output_text.split("[/INST]")
        output_text = output_text_list[-1].strip()

    return output_text