import json
import argparse
import sys
from ChatDrug.task_and_evaluation.Conversational_LLMs_utils import complete
from utils import (
    construct_PDDS_prompt, construct_PDDS_prompt_galactica, load_retrieval_DB, fast_protein_dict,
    retrieve_and_feedback, retrieve_and_feedback_fast_protein, load_thredhold
)
from ChatDrug.task_and_evaluation import task_to_drug, get_task_specification_dict, evaluate, parse
from ChatDrug.task_and_evaluation.protein_editing import evaluate_fast_protein
from transformers import LlamaTokenizer, LlamaForCausalLM, AutoTokenizer, OPTForCausalLM

def conversation(messages, model, tokenizer, conversational_LLM, C, round_index, trial_index, task, 
        drug_type, input_drug, retrieval_DB, record, logfile, fast_protein, constraint, 
        threshold_dict, sim_DB_dict, test_example_dict):
    generated_text = complete(messages, model, tokenizer, conversational_LLM, drug_type, round_index)
    messages.append({"role": "assistant", "content": generated_text})
    
    print("----------------", file=logfile)
    print("User:" + messages[2*round_index+1]["content"], file=logfile)
    print("AI:" + generated_text, file=logfile)
    
    record[input_drug]['retrieval_conversation'][round_index]['user'] = messages[2*round_index +1]["content"]
    record[input_drug]['retrieval_conversation'][round_index]['chatgpt'] = generated_text

    if round_index >= 1:
        closest_drug = record[input_drug]['retrieval_conversation'][round_index-1]['retrieval_drug']
    else:
        closest_drug = None
    generated_drug_list = parse(task, input_drug, generated_text, closest_drug)

    # Check Parsing Results
    if generated_drug_list == None:
        record[input_drug]['skip_round'] = round_index
        return -1, None
    elif len(generated_drug_list) == 0:
        record[input_drug]['retrieval_conversation'][round_index]['answer'] = 'False'
        return 0, None
    else:
        generated_drug = generated_drug_list[:min(len(generated_drug_list),5)][trial_index]
        print("Generated Result:"+str(generated_drug), file=logfile)
        record[input_drug]['retrieval_conversation'][round_index]['generated_drug'] = generated_drug
    
    # Check Evaluation Results
    if drug_type == 'protein' and fast_protein:
        answer = evaluate_fast_protein([input_drug], [generated_drug], task, test_example_dict)
        answer = answer[0]
    else:
        answer = evaluate(input_drug, generated_drug, task, constraint, threshold_dict)

    if answer == -1:
        record[input_drug]['skip_round'] = round_index
        return -1, None

    print('Evaluation result is: '+str(answer), file=logfile)
    record[input_drug]['retrieval_conversation'][round_index]['answer']=str(answer)

    if answer:
        return 1, generated_drug
    else:
        if round_index < C:
            answer, generated_drug  = ReDF(messages, conversational_LLM, round_index, task, drug_type, input_drug, generated_drug, 
                retrieval_DB, record, logfile, fast_protein, constraint, threshold_dict, sim_DB_dict, test_example_dict)
        return answer, generated_drug


def ReDF(messages, conversational_LLM, round_index, task, drug_type, input_drug, generated_drug, 
        retrieval_DB, record, logfile, fast_protein, constraint, threshold_dict, sim_DB_dict, test_example_dict):
    print(f'Start Retrieval {round_index+1}', file=logfile)
    try:
        if drug_type=='protein' and fast_protein:
            closest_drug = retrieve_and_feedback_fast_protein(task, sim_DB_dict, test_example_dict, input_drug, generated_drug)
        else:
            closest_drug = retrieve_and_feedback(task, retrieval_DB, input_drug, generated_drug, constraint, threshold_dict)
    except:
        error = sys.exc_info()
        if error[0] == Exception:
            print('Cannot find a retrieval result.', file=logfile)
            return 0, None
        else:
            print('Invalid drug. Failed to evaluate. Skipped.', file=logfile)
            record[input_drug]['skip_round'] = round_index
            return -1, None

    print("Retrieval Result:" + closest_drug, file=logfile)
    record[input_drug]['retrieval_conversation'][round_index]['retrieval_drug'] = closest_drug

    if conversational_LLM=="galactica":
        if drug_type=="molecule":
            prompt_ReDF = f'Question: Your provided sequence [START_I_SMILES]{generated_drug}[END_I_SMILES] is not correct. We find a sequence [START_I_SMILES]{closest_drug}[END_I_SMILES] which is correct and similar to the {drug_type} you provided. Can you give me a new {drug_type}?\n\nAnswer:'
        elif drug_type=="peptide":
            prompt_ReDF = f'Question: Your provided sequence {generated_drug} is not correct. We find a sequence {closest_drug} which is correct and similar to the {drug_type} you provided. Can you give me a new {drug_type}?\n\nAnswer:'
        elif drug_type=="protein":
            prompt_ReDF = f'Question: Your provided sequence [START_AMINO]{generated_drug}[END_AMINO] is not correct. We find a sequence [START_AMINO]{closest_drug}[END_AMINO] which is correct and similar to the {drug_type} you provided. Can you give me a new {drug_type}?\n\nAnswer:'
    else:
        prompt_ReDF = f'Your provided sequence {generated_drug} is not correct. We find a sequence {closest_drug} which is correct and similar to the {drug_type} you provided. Can you give me a new {drug_type}?'
    messages.append({"role": "user", "content": prompt_ReDF})

    return 0, generated_drug


def main(args):
    f = open(args['log_file'], 'w')
    record = {}

    # load dataset
    drug_type = task_to_drug(args['task'])
    task_specification_dict = get_task_specification_dict(args['task'])
    input_drug_list, retrieval_DB = load_retrieval_DB(args['task'], args['seed'])
    threshold_dict = load_thredhold(drug_type)
    sim_DB_dict, test_example_dict = fast_protein_dict(args['task'], drug_type, args['fast_protein'], input_drug_list, retrieval_DB)

    # load model
    if args['conversational_LLM'] == 'llama2':
        model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", return_dict=True,
            load_in_8bit=False, device_map="auto", low_cpu_mem_usage=True)
        tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
        tokenizer.add_special_tokens({"pad_token": "<PAD>"})
    elif args['conversational_LLM'] == 'galactica':
        model = OPTForCausalLM.from_pretrained("facebook/galactica-6.7b", device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained("facebook/galactica-6.7b")
    else: 
        model = None
        tokenizer = None

    num_correct = 0
    num_all = 0

    for index, input_drug in enumerate(input_drug_list):
        print(f">>Sample {index}", file=f)
        
        # init record[input_drug]
        record[input_drug]={}
        record[input_drug]['skip_conversation_round'] = -1
        record[input_drug]['retrieval_conversation'] = [{'result':i} for i in range((args['C']+1))]

        # ChatGPT message
        messages = [{"role": "system", "content": "You are an expert in the field of molecular chemistry."}]

        # PDDS
        if args['conversational_LLM'] == 'galactica':
            PDDS_prompt = construct_PDDS_prompt_galactica(task_specification_dict, input_drug, drug_type, args['task'])
        else:
            PDDS_prompt = construct_PDDS_prompt(task_specification_dict, input_drug, drug_type, args['task'])
        messages.append({"role": "user", "content": PDDS_prompt})

        for round_index in range((args['C']+1)):
            answer, output_drug = conversation(
                messages=messages, model=model, tokenizer=tokenizer, conversational_LLM=args['conversational_LLM'],
                C=args['C'], round_index=round_index, trial_index=args['trial_index'], task=args['task'],
                drug_type=drug_type, input_drug=input_drug, retrieval_DB=retrieval_DB, record=record,
                logfile=f, fast_protein=args['fast_protein'], constraint=args['constraint'], 
                threshold_dict=threshold_dict, sim_DB_dict=sim_DB_dict, test_example_dict=test_example_dict)
           
            if answer != 0 or output_drug == None:
                break         
        
        if answer == -1:
            continue
        elif answer:
            num_correct += 1
            num_all += 1
        else:
            num_all += 1

        print(f'Acc = {num_correct}/{num_all}', file=f)
        print("----------------", file=f)

    print("--------Final Acc--------", file=f)
    print(f'Acc = {num_correct}/{num_all}', file=f)
    print("----------------", file=f)

    with open(args['record_file'], 'w') as rf:
        json.dump(record, rf, ensure_ascii=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # required arguments
    parser.add_argument('--task', action='store', required=True, type=int, help='task_id')
    parser.add_argument('--conversational_LLM', action='store', required=False, type=str, default='chatgpt', help='chatgpt, llama2, galatica')
    parser.add_argument('--log_file', action='store', required=False, type=str, default='results/ChatDrug.log', help='saved log file name')
    parser.add_argument('--record_file', action='store', required=False, type=str, default='results/ChatDrug.json', help='saved record file name')
    parser.add_argument('--constraint', required=False, type=str, default='loose', help='loose or strict')
    parser.add_argument('--seed', required=False, type=int, default=0, help='seed for retrieval data base')
    parser.add_argument('--trial_index', required=False, type=int, default=0, help='trial index for molecule editing tasks')
    parser.add_argument('--C', required=False, type=int, default=2, help='number of conversation round')
    parser.add_argument('--fast_protein', required=False, type=bool, default=True, help='whether to use fast protein evaluation')
    args = parser.parse_args()
    args = vars(args)

    main(args)