import numpy as np
from .small_molecule_editing import evaluate_molecule, task_specification_dict_molecule, parse_molecule, task2threshold_list
from .peptide_editing import evaluate_peptide, task_specification_dict_peptide, parse_peptide
from .protein_editing import evaluate_pairwise_list_result, load_ProteinDT_model, task_specification_dict_protein, parse_protein
from transformers import BertTokenizerFast

def task_to_drug(task):
    if task < 300:
        return 'molecule'
    elif task < 500:
        return 'peptide'
    elif task < 600:
        return 'protein'
    else:
        raise NotImplementedError
    

def get_task_specification_dict(task):
    if task < 300:
        return task_specification_dict_molecule
    elif task < 500:
        return task_specification_dict_peptide
    elif task < 600:
        return task_specification_dict_protein
    else:
        raise NotImplementedError


def parse(task, input_drug, generated_text, addition_drug=None):
    if task < 300:
        return parse_molecule(input_drug, generated_text, addition_drug)
    elif task < 500:
        return parse_peptide(input_drug, generated_text, addition_drug)
    elif task < 600:
        return parse_protein(input_drug, generated_text, addition_drug)
    else:
        raise NotImplementedError


def evaluate(input_drug, generated_drug, task, constraint, threshold_dict):
    if task<300:
        if constraint == 'loose':
            threshold_list = task2threshold_list[task][0]
        else:
            threshold_list = task2threshold_list[task][1]
        _, _, answer = evaluate_molecule(input_drug, generated_drug, task, threshold_list=threshold_list)
    elif task<400:
        task_specification_dict = get_task_specification_dict('peptide')
        _, _, target_allele_type = task_specification_dict[task]
        try:
            _, _, answer = evaluate_peptide([input_drug], [generated_drug], target_allele_type, [threshold_dict[target_allele_type]])
        except:
            return -1
        answer = answer[0]
    elif task<500:
        task_specification_dict = get_task_specification_dict('peptide')
        _, _, target_allele_type1, target_allele_type2 = task_specification_dict[task]
        try:
            _, _, answer1 = evaluate_peptide([input_drug], [generated_drug], target_allele_type1, [threshold_dict[target_allele_type1]])
            _, _, answer2 = evaluate_peptide([input_drug], [generated_drug], target_allele_type2, [threshold_dict[target_allele_type2]])
        except:
            return -1
        answer = np.logical_and(answer1, answer2)
        answer = answer[0]
    else:
        device = "cuda"
        chache_dir = "./data/protein_editing/temp_pretrained_ProteinDT"
        input_model_path = "./data/protein_editing/pytorch_model_ss3.bin"
        model = load_ProteinDT_model(input_model_path, chache_dir, mean_output=True, num_labels=3)
        model = model.to(device)
        tokenizer = BertTokenizerFast.from_pretrained("Rostlab/prot_bert_bfd", chache_dir=chache_dir, do_lower_case=False)
        _, _, answer = evaluate_pairwise_list_result(model=model, tokenizer=tokenizer, input_protein_list=[input_drug], output_protein_list=[generated_drug], task_id=task, device=device)
        answer = answer[0]

    return answer
