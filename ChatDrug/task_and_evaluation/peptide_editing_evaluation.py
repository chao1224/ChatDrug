import numpy as np
from mhcflurry import Class1PresentationPredictor


model_pretrained_checkpoint = "../data/peptide/models_class1_presentation/models"
MHC_peptide_predictor = Class1PresentationPredictor.load(model_pretrained_checkpoint)

EPS = 1e-10


def evaluate_result(input_peptide_sequence_list, output_peptide_sequence_list, target_allele_type, threshold_list=[0.75]):
    input_df = MHC_peptide_predictor.predict(peptides=input_peptide_sequence_list, alleles=[target_allele_type], verbose=False)
    input_value = input_df["presentation_score"].to_list()
    input_value = np.array(input_value)

    output_df = MHC_peptide_predictor.predict(peptides=output_peptide_sequence_list, alleles=[target_allele_type], verbose=False)
    output_value = output_df["presentation_score"].to_list()
    output_value = np.array(output_value)

    flag = np.logical_and((output_value > input_value + EPS), (output_value > threshold_list[0]))

    return input_df, output_df, flag
