from ChatDrug.task_and_evaluation.peptide_editing import load_allele2protein_sequence, load_selected_allele_list, load_raw_allele2peptide
import json
import random
from itertools import permutations
import numpy as np
from mhcflurry import Class1PresentationPredictor
model_pretrained_checkpoint = "./models_class1_presentation/models"
MHC_peptide_predictor = Class1PresentationPredictor.load(model_pretrained_checkpoint)

invalid_peptide_set = ("EVIGVTXLM")


if __name__ == "__main__":
    random.seed(42)

    allele_sequence_file = "class1_pseudosequences.csv"
    allele2protein_sequence = load_allele2protein_sequence(allele_sequence_file)
    print("allele2protein_sequence", len(allele2protein_sequence))

    selected_allele_file = "selected_alleles.txt"
    selected_allele_list = load_selected_allele_list(selected_allele_file)
    print("selected_allele_list", len(selected_allele_list))
    print(selected_allele_list)

    peptide_file = "Data_S3.csv"
    allele2peptide = load_raw_allele2peptide(peptide_file)
    print("len allele2peptide", len(allele2peptide))

    TARGET_LEN = 9
    selected_allele2peptide = {}
    for allele in selected_allele_list:
        peptide_list = allele2peptide[allele]

        len_list = [len(x) for x in peptide_list]
        majority_len = int(np.median(len_list))

        if majority_len != TARGET_LEN:
            print("invalid majority len for {} ({})".format(allele, majority_len))
            continue

        neo_peptide_list = []
        for x in peptide_list:
            if len(x) != majority_len:
                continue
            if x in invalid_peptide_set:
                continue
            neo_peptide_list.append(x)

        random.shuffle(neo_peptide_list)
        # only keep the first 1K peptides
        # selected_peptide_list = neo_peptide_list[:500]
        selected_peptide_list = []
        for x in neo_peptide_list:
            if x not in selected_peptide_list:
                selected_peptide_list.append(x)
            else:
                print("HIT")
            if len(selected_peptide_list) >= 300:
                break
        selected_allele2peptide[allele] = selected_peptide_list
        print("selected allele: {}\tlen of peptides: {}\tlen of peptides with length {}: {}".format(allele, len(peptide_list), majority_len, len(neo_peptide_list)))
    
    f = open("peptide_editing.json", "w")
    json.dump(selected_allele2peptide, f)

    selected_allele_affinity_threshold = {}
    for allele, selected_peptide_list in selected_allele2peptide.items():
        predict_df = MHC_peptide_predictor.predict(peptides=selected_peptide_list, alleles=[allele], verbose=False)
        print(predict_df)
        predict_affinity = predict_df["presentation_score"].to_list()
        predict_affinity = np.array(predict_affinity)
        mean_predict_affinity = np.mean(predict_affinity)
        selected_allele_affinity_threshold[allele] = mean_predict_affinity
        print("predict_affinity", predict_affinity)
        print("mean predict_affinity", mean_predict_affinity)
    f = open("peptide_editing_threshold.json", "w")
    json.dump(selected_allele_affinity_threshold, f)
