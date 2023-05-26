from collections import defaultdict

AMINO_ACIDS = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]


task_specification_dict = {
    301: [
        "We want a peptide that binds to TARGET_ALLELE_TYPE. We have a peptide PEPTIDE_SEQUENCE that binds to SOURCE_ALLELE_TYPE, can you help modify it? The output peptide should be similar to input peptide.",
        "HLA-C*16:01", "HLA-B*44:02"
    ],
    302: [
        "We want a peptide that binds to TARGET_ALLELE_TYPE. We have a peptide PEPTIDE_SEQUENCE that binds to SOURCE_ALLELE_TYPE, can you help modify it? The output peptide should be similar to input peptide.",
        "HLA-B*08:01", "HLA-C*03:03"
    ],
    303: [
        "We want a peptide that binds to TARGET_ALLELE_TYPE. We have a peptide PEPTIDE_SEQUENCE that binds to SOURCE_ALLELE_TYPE, can you help modify it? The output peptide should be similar to input peptide.",
        "HLA-C*12:02", "HLA-B*40:01"
    ],
    304: [
        "We want a peptide that binds to TARGET_ALLELE_TYPE. We have a peptide PEPTIDE_SEQUENCE that binds to SOURCE_ALLELE_TYPE, can you help modify it? The output peptide should be similar to input peptide.",
        "HLA-A*11:01", "HLA-B*08:01"
    ],
    305: [
        "We want a peptide that binds to TARGET_ALLELE_TYPE. We have a peptide PEPTIDE_SEQUENCE that binds to SOURCE_ALLELE_TYPE, can you help modify it? The output peptide should be similar to input peptide.",
        "HLA-A*24:02", "HLA-B*08:01"
    ],
    306: [
        "We want a peptide that binds to TARGET_ALLELE_TYPE. We have a peptide PEPTIDE_SEQUENCE that binds to SOURCE_ALLELE_TYPE, can you help modify it? The output peptide should be similar to input peptide.",
        "HLA-C*12:02", "HLA-B*40:02"
    ],


    401: [
        "We want a peptide that binds to TARGET_ALLELE_TYPE_01 and TARGET_ALLELE_TYPE_02. We have a peptide PEPTIDE_SEQUENCE that binds to SOURCE_ALLELE_TYPE, can you help modify it? The output peptide should be similar to input peptide.",
        "HLA-A*29:02", "HLA-B*08:01", "HLA-C*15:02"
    ],
    402: [
        "We want a peptide that binds to TARGET_ALLELE_TYPE_01 and TARGET_ALLELE_TYPE_02. We have a peptide PEPTIDE_SEQUENCE that binds to SOURCE_ALLELE_TYPE, can you help modify it? The output peptide should be similar to input peptide.",
        "HLA-A*03:01", "HLA-B*40:02", "HLA-C*14:02"
    ],
    403: [
        "We want a peptide that binds to TARGET_ALLELE_TYPE_01 and TARGET_ALLELE_TYPE_02. We have a peptide PEPTIDE_SEQUENCE that binds to SOURCE_ALLELE_TYPE, can you help modify it? The output peptide should be similar to input peptide.",
        "HLA-C*14:02", "HLA-B*08:01", "HLA-A*11:01"
    ],
}


def load_allele2protein_sequence(file_path):
    f = open(file_path, "r")
    allele2protein_sequence = {}
    for line in f.readlines()[1:]:
        line = line.strip()
        line = line.split(" ")
        allele = line[0]
        protein_sequence = line[1]
        if allele in allele2protein_sequence:
            continue
        allele2protein_sequence[allele] = protein_sequence
    return allele2protein_sequence


def load_selected_allele_list(file_path):
    f = open(file_path, "r")
    allele_list = []
    for line in f.readlines():
        allele_list.append(line.strip())
    return allele_list


def load_raw_allele2peptide(file_path):
    import pandas as pd
    df = pd.read_csv(file_path)
    allele_list = df["allele"].tolist()
    peptide_list = df["peptide"].tolist()
    allele2peptide = defaultdict(list)
    
    for allele, peptide in zip(allele_list, peptide_list):
        allele2peptide[allele].append(peptide)
    return allele2peptide


def load_processed_allele2peptide(file_path):
    import json
    f = open(file_path, "r")
    data = json.load(f)
    return data
