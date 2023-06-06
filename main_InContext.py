import json
import argparse
import sys
from ChatDrug.task_and_evaluation.Conversational_LLMs_utils import complete
from utils import load_retrieval_DB, construct_prompt_incontext, retrieve_and_feedback, load_thredhold
from ChatDrug.task_and_evaluation import task_to_drug, get_task_specification_dict, evaluate, parse


def main(args):
    f = open(args['log_file'], 'w')
    record = {}

    # load dataset
    drug_type = task_to_drug(args['task'])
    task_specification_dict = get_task_specification_dict(args['task'])
    input_drug_list, retrieval_DB = load_retrieval_DB(args['task'], args['seed'])
    threshold_dict = load_thredhold(drug_type)

    num_correct = 0
    num_all = 0
    num_skip = 0

    for index, input_drug in enumerate(input_drug_list):
        print(f">>Sample {index}", f)

        record[input_drug]={}
        record[input_drug]['drug_skip'] = 0
        
        # ChatGPT message
        messages = [{"role": "system", "content": "You are an expert in the field of molecular chemistry."}]

        print(f'Start Retrieval', f)
        try:
            closest_drug = retrieve_and_feedback(args['task'], retrieval_DB, input_drug, input_drug, args['constraint'], threshold_dict)
        except:
            error = sys.exc_info()
            if error[0] == Exception:
                print('Cannot find a retrieval result.', f)
                record[input_drug]['answer'] = 'False'
                num_all += 1
            else:
                print('Invalid drug. Failed to evaluate. Skipped.', f)
                record[input_drug]['drug_skip'] = 1
                num_skip += 1
            continue
            
        print("Retrieval Result:" + closest_drug, f)
        record[input_drug]['retrieval_drug'] = closest_drug

        prompt = construct_prompt_incontext(task_specification_dict, input_drug, drug_type, closest_drug, args['task'])
        messages.append({"role": "user", "content": prompt})

        generated_text = complete(messages, args['conversational_LLM'])
        messages.append({"role": "assistant", "content": generated_text})

        print("----------------", f)
        print("User:" + prompt, f)
        print("ChatGPT:" + generated_text, f)
        record[input_drug]['user'] = prompt
        record[input_drug]['chatgpt'] = generated_text

        generated_drug_list = parse(args['task'], input_drug, generated_text, closest_drug)
        # Check Parsing Results
        if generated_drug_list == None:
            record[input_drug]['drug_skip'] = 1
            num_skip += 1
            continue
        elif len(generated_drug_list) == 0:
            record[input_drug]['answer'] = 'False'
            num_all += 1
            continue
        else:
            generated_drug = generated_drug_list[0]
            print("Generated Result:" + str(generated_drug), f)
            record[input_drug]['generated_drug'] = generated_drug

        answer = evaluate(input_drug, generated_drug, args['task'], args['constraint'], threshold_dict)

        if answer == -1:
            record[input_drug]['drug_skip'] = 1
            num_skip += 1
            continue

        print('Evaluation result is: ' + str(answer), f)
        record[input_drug]['answer'] = str(answer)

        if answer:
            num_correct += 1
            num_all += 1
        else:
            num_all += 1

        print(f'Acc = {num_correct}/{(num_all-num_skip)}', f)
        print("----------------", f)

    print("--------Final Acc--------", f)
    print(f'Acc = {num_correct}/{(num_all-num_skip)}', f)
    print("----------------", f)

    with open(args['record_file'], 'w') as rf:
        json.dump(record, rf, ensure_ascii=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # required arguments
    parser.add_argument('--task', action='store', required=True, help='task_id')
    parser.add_argument('--conversational_LLM', action='store', required=False, type=str, default='chatgpt', help='only support chatgpt now')
    parser.add_argument('--log_file', action='store', required=False, type=str, default='results/ChatDrug.log', help='saved log file name')
    parser.add_argument('--record_file', action='store', required=False, type=str, default='results/ChatDrug.json', help='saved record file name')
    parser.add_argument('--constraint', required=False, type=str, default='loose', help='loose or strict')
    parser.add_argument('--seed', required=False, type=int, default=0, help='seed for retrieval data base')
    args = parser.parse_args()
    args = vars(args)

    main(args)