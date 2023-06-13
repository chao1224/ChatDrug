import os
import json
import numpy as np
import pandas as pd

from rdkit.Chem import Draw, AllChem
from rdkit import Chem, DataStructs
import Levenshtein

from ChatDrug.task_and_evaluation import get_task_specification_dict, evaluate
from ChatDrug.task_and_evaluation.protein_editing import ProteinSecondaryStructureDataset, load_ProteinDT_model, evaluate_fast_protein_dict
from transformers import BertTokenizerFast


def construct_PDDS_prompt(task_specification_dict, input_drug, drug_type, task):
    if drug_type == 'molecule':
        task_prompt_template = task_specification_dict[task]
        prompt = task_prompt_template.replace('SMILES_PLACEHOLDER', input_drug)
        prompt = prompt + " Give me five molecules in SMILES only and list them using bullet points. No explanation is needed." 
    if drug_type == 'peptide':
        if task < 400: 
            task_prompt_template, source_allele_type, target_allele_type = task_specification_dict[task]
            prompt = task_prompt_template.replace("SOURCE_ALLELE_TYPE", source_allele_type).replace("TARGET_ALLELE_TYPE", target_allele_type).replace("PEPTIDE_SEQUENCE", input_drug)
        else: 
            task_prompt_template, source_allele_type, target_allele_type1, target_allele_type2 = task_specification_dict[task]
            prompt = task_prompt_template.replace("SOURCE_ALLELE_TYPE", source_allele_type).replace("TARGET_ALLELE_TYPE_01", target_allele_type1).replace("TARGET_ALLELE_TYPE_02", target_allele_type2).replace("PEPTIDE_SEQUENCE", input_drug)
        prompt = prompt + ' Please provide the possible modified peptide sequence only. No explanation is needed.'
    if drug_type == 'protein':
        task_prompt_template = task_specification_dict[task]
        prompt = task_prompt_template.replace("PROTEIN_SEQUENCE_PLACEHOLDER", input_drug)
        prompt = prompt + ' No explanation is needed.'
    return prompt


def construct_prompt_incontext(task_specification_dict, input_drug, drug_type, closest_smiles, task):
    if drug_type == 'molecule':
        task_prompt_template = task_specification_dict[task]
        prompt = task_prompt_template.replace('SMILES_PLACEHOLDER', input_drug)
        prompt = prompt + " We have known that " + f'similar {drug_type} {closest_smiles} is one of the correct answers. ' + "Give me another five molecules in SMILES only and list them using bullet points. No explanation is needed."
    if drug_type == 'peptide':
        if task < 400: 
            task_prompt_template, source_allele_type, target_allele_type = task_specification_dict[task]
            prompt = task_prompt_template.replace("SOURCE_ALLELE_TYPE", source_allele_type).replace("TARGET_ALLELE_TYPE", target_allele_type).replace("PEPTIDE_SEQUENCE", input_drug)
        else: 
            task_prompt_template, source_allele_type, target_allele_type1, target_allele_type2 = task_specification_dict[task]
            prompt = task_prompt_template.replace("SOURCE_ALLELE_TYPE", source_allele_type).replace("TARGET_ALLELE_TYPE_01", target_allele_type1).replace("TARGET_ALLELE_TYPE_02", target_allele_type2).replace("PEPTIDE_SEQUENCE", input_drug)
        prompt = prompt + " We have known that " + f'similar {drug_type} {closest_smiles} is one of the correct answers. ' + "Please provide another possible modified peptide sequence only. No explanation is needed."
    if drug_type == 'protein':
        task_prompt_template = task_specification_dict[task]
        prompt = task_prompt_template.replace("PROTEIN_SEQUENCE_PLACEHOLDER", input_drug)
        prompt = prompt + " We have known that " + f'similar {drug_type} {closest_smiles} is one of the correct answers. ' + "Please provide another possible modified protein only. No explanation is needed."
    return prompt 


def load_dataset(drug_type, task, task_specification_dict):
    if drug_type == 'molecule':
        with open('data/small_molecule/small_molecule_editing.txt') as f:
            test_data = f.read().splitlines()
    elif drug_type == 'peptide':
        if task < 400: 
            _, source_allele_type, _ = task_specification_dict[task]
        else:
            _, source_allele_type, _, _ = task_specification_dict[task]
        f = open("data/peptide_editing/peptide_editing.json", "r")
        data = json.load(f)
        test_data = data[source_allele_type]
    elif drug_type == 'protein':
        data_dir = "./data/protein_editing/downstream_datasets"
        chache_dir = "./data/protein_editing/temp_pretrained_ProteinDT"
        data_file = os.path.join(data_dir, "secondary_structure", "secondary_structure_cb513.lmdb")
        tokenizer = BertTokenizerFast.from_pretrained("Rostlab/prot_bert_bfd", chache_dir=chache_dir, do_lower_case=False)
        dataset = ProteinSecondaryStructureDataset(data_file, tokenizer)
        test_data = dataset.protein_sequence_list
    elif drug_type == 'retrieval':
        data_dir = "./data/protein_editing/downstream_datasets"
        chache_dir = "./data/protein_editing/temp_pretrained_ProteinDT"
        data_file = os.path.join(data_dir, "secondary_structure", "secondary_structure_train.lmdb")
        tokenizer = BertTokenizerFast.from_pretrained("Rostlab/prot_bert_bfd", chache_dir=chache_dir, do_lower_case=False)
        dataset = ProteinSecondaryStructureDataset(data_file, tokenizer)
        test_data = dataset.protein_sequence_list
    else:
        raise NotImplementedError
    return test_data


def load_retrieval_DB(task, seed):
    if task < 300:
        drug_type = 'molecule'
        DBfile = 'data/small_molecule/250k_rndm_zinc_drugs_clean_3.csv'
        task_specification_dict = get_task_specification_dict(task)
        input_drug_list = load_dataset(drug_type, task, task_specification_dict)
        input_drug_list = list(set(input_drug_list))
        DB = pd.read_csv(DBfile)
        DB = DB[['smiles']]
        DB = DB.rename(columns={"smiles": "sequence"})

        for SEQUENCE_TO_BE_MODIFIED in input_drug_list:
            DB = DB[DB['sequence'].str.find(SEQUENCE_TO_BE_MODIFIED)<0]
        DB=DB.sample(10000, random_state=seed)

    elif task<400:
        drug_type = 'peptide'
        DBfile = 'data/peptide_editing/Data_S3.csv'

        task_specification_dict = get_task_specification_dict(task)
        target_allele_type = task_specification_dict[task][2]
        input_drug_list = load_dataset(drug_type, task, task_specification_dict)
        input_drug_list = list(set(input_drug_list))
        DB = pd.read_csv(DBfile)
        DB = DB[DB['allele']==target_allele_type]
        DB = DB[['peptide']]
        DB = DB.rename(columns={"peptide": "sequence"})
        # remove duplication
        for SEQUENCE_TO_BE_MODIFIED in input_drug_list:
            DB = DB[DB['sequence'].str.find(SEQUENCE_TO_BE_MODIFIED)<0]

    elif task<500:
        drug_type = 'peptide'
        DBfile = 'data/peptide_editing/Data_S3.csv'

        task_specification_dict = get_task_specification_dict(task)
        target_allele_type1 = task_specification_dict[task][2]
        target_allele_type2 = task_specification_dict[task][3]
        input_drug_list = load_dataset(drug_type, task, task_specification_dict)
        input_drug_list = list(set(input_drug_list))
        DB = pd.read_csv(DBfile)
        DB1 = DB[DB['allele']==target_allele_type1]
        DB1 = DB1[['peptide']]

        DB2 = DB[DB['allele']==target_allele_type2]
        DB2 = DB2[['peptide']]

        DB = pd.concat([DB1, DB2])
        DB.drop_duplicates(subset=['peptide'],keep='first',inplace=True)

        DB = DB.rename(columns={"peptide": "sequence"})
        # remove duplication
        for SEQUENCE_TO_BE_MODIFIED in input_drug_list:
            DB = DB[DB['sequence'].str.find(SEQUENCE_TO_BE_MODIFIED)<0]

    else:
        drug_type = 'protein'
        task_specification_dict = get_task_specification_dict(task)
        input_drug_list = load_dataset(drug_type, task, task_specification_dict)
        input_drug_list = list(set(input_drug_list))
        DBfile = load_dataset('retrieval', task, task_specification_dict)
        DB = pd.DataFrame(DBfile)
        DB = DB.rename(columns={0: "sequence"})
        # remove duplication
        for SEQUENCE_TO_BE_MODIFIED in input_drug_list:
            DB = DB[DB['sequence'].str.find(SEQUENCE_TO_BE_MODIFIED)<0]

    return input_drug_list, DB


def load_thredhold(drug_type):
    if drug_type == 'peptide':
        f_threshold = open("data/peptide_editing/peptide_editing_threshold.json", 'r')
        threshold_dict = json.load(f_threshold)
        for k, v in threshold_dict.items():
            threshold_dict[k] = v/2
        f_threshold.close()
    else:
        threshold_dict = None
    return threshold_dict


def sim_molecule(smile0, smile1):
    mol0 = Chem.MolFromSmiles(smile0)
    fp0 = AllChem.GetMorganFingerprint(mol0, 2)

    mol1 = Chem.MolFromSmiles(smile1)
    fp1 = AllChem.GetMorganFingerprint(mol1, 2)
    
    sim = DataStructs.TanimotoSimilarity(fp0, fp1)
    return sim


def sim_sequence(task, SEQ1, SEQ2):
    if task<300:
        sim = sim_molecule(SEQ1,SEQ2)
    else:
        sim = Levenshtein.distance(SEQ1,SEQ2)
    return sim


def retrieve_and_feedback(task, DB, input_drug, generated_drug, constraint, threshold_dict):
    sim_DB = DB.copy()
    # sim_DB['sim']=[0]*len(DB)
    sim_list = []
    for index, row in sim_DB.iterrows():
        smiles = row['sequence'].replace('\n','')
        sim = sim_sequence(task, smiles, generated_drug)
        sim_list.append(sim)
        # sim_DB['sim'][index] = sim
        # sim_DB.at[index,'sim']=sim
    sim_DB['sim'] = sim_list
    if task<300:
        sim_DB = sim_DB.sort_values(by=['sim'],ascending=False)
    else:
        sim_DB = sim_DB.sort_values(by=['sim'])

    for index, row in sim_DB.iterrows():
        answer = evaluate(input_drug, row['sequence'].replace('\n',''), task, constraint, threshold_dict=threshold_dict)
        if answer:
            return row['sequence'].replace('\n','')
    raise Exception("Sorry, Cannot fined a good one")


def retrieve_and_feedback_fast_protein(task, sim_DB_dict, test_example_dict, input_drug, generated_drug):
    sim_DB = list(sim_DB_dict.keys())
    sim_DB_dict_output = {}
    for instance in sim_DB:
        sim = sim_sequence(task, instance, generated_drug)
        sim_DB_dict_output[instance] = sim

    sim_DB_list = sorted(sim_DB_dict_output.items(), key = lambda kv:(kv[1], kv[0]))

    for output_sequence, sim in sim_DB_list:
        answer = [sim_DB_dict[output_sequence]]>[test_example_dict[input_drug]]
        if answer:
            return output_sequence
    raise Exception("Sorry, Cannot fined a good one")


def generate_retrieval_dict(task, input_drug_list, DB, saved_file):
    sim_DB = DB.copy()
    sim_DB_list = sim_DB['sequence'].tolist()

    device = "cuda"
    chache_dir = "./data/protein_editing/temp_pretrained_ProteinDT"
    input_model_path = "./data/protein_editing/pytorch_model_ss3.bin"
    model = load_ProteinDT_model(input_model_path, chache_dir, mean_output=True, num_labels=3)
    model = model.to(device)
    tokenizer = BertTokenizerFast.from_pretrained("Rostlab/prot_bert_bfd", chache_dir=chache_dir, do_lower_case=False)

    input_count_list = evaluate_fast_protein_dict(model=model, tokenizer=tokenizer, input_protein_list=sim_DB_list, task_id=task, device=device)
    sim_DB_dict = dict(zip(sim_DB_list, input_count_list))
    np.save(saved_file + '/sim_DB_dict_'+str(task)+'.npy', sim_DB_dict)

    test_example_list = []
    for input_drug in input_drug_list:
        test_example_list.append(input_drug)
    test_count_list = evaluate_fast_protein_dict(model=model, tokenizer=tokenizer, input_protein_list=test_example_list, task_id=task, device=device)
    test_example_dict = dict(zip(test_example_list, test_count_list))
    np.save(saved_file + '/test_example_dict_'+str(task)+'.npy', test_example_dict)
    return sim_DB_dict, test_example_dict


def fast_protein_dict(task, drug_type, fast_protein, input_drug_list, retrieval_DB):
    if drug_type=='protein' and fast_protein:
        sim_DB_dict, test_example_dict = generate_retrieval_dict(task, input_drug_list, retrieval_DB, './saved_fast_protein_dict')
    else:
        sim_DB_dict, test_example_dict = None, None
    return sim_DB_dict, test_example_dict