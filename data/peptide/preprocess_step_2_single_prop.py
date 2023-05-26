from ChatDrug.task_and_evaluation.peptide_editing import load_allele2protein_sequence, load_selected_allele_list, load_raw_allele2peptide
import json
import random
from itertools import permutations


if __name__ == "__main__":
    random.seed(42)

    allele_sequence_file = "class1_pseudosequences.csv"
    allele2protein_sequence = load_allele2protein_sequence(allele_sequence_file)
    print("allele2protein_sequence", len(allele2protein_sequence))

    selected_allele_file = "selected_alleles.txt"
    selected_allele_list = load_selected_allele_list(selected_allele_file)
    selected_allele_list.remove("HLA-A*68:01")
    selected_allele_list.remove("HLA-B*27:05")
    selected_allele_list.remove("HLA-B*57:01")
    print("selected_allele_list", len(selected_allele_list))
    print(selected_allele_list)

    permutation_list = list(permutations(selected_allele_list, 2))
    random.shuffle(permutation_list)
    valid_count = 0
    for perm_ in permutation_list:
        source_allele, target_allele = perm_
        if source_allele[:5] != target_allele[:5]:
            print(valid_count)
            print("\"{}\", \"{}\"".format(source_allele, target_allele))
            valid_count += 1
            if valid_count >= 10:
                break
