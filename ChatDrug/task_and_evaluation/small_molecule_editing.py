from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

props = ["MolLogP", "qed", "TPSA", "NumHAcceptors", "NumHDonors"]
prop_pred = [(n, func) for n, func in Descriptors.descList if n.split("_")[-1] in props]

prop2func = {}
for prop, func in prop_pred:
    prop2func[prop] = func


task_specification_dict = {
    101: "Can you make molecule SMILES_PLACEHOLDER more soluble in water? The output molecule should be similar to the input molecule.",
    102: "Can you make molecule SMILES_PLACEHOLDER less soluble in water? The output molecule should be similar to the input molecule.",
    103: "Can you make molecule SMILES_PLACEHOLDER more like a drug? The output molecule should be similar to the input molecule.",
    104: "Can you make molecule SMILES_PLACEHOLDER less like a drug? The output molecule should be similar to the input molecule.",
    105: "Can you make molecule SMILES_PLACEHOLDER higher permeability? The output molecule should be similar to the input molecule.",
    106: "Can you make molecule SMILES_PLACEHOLDER lower permeability? The output molecule should be similar to the input molecule.",
    107: "Can you make molecule SMILES_PLACEHOLDER with more hydrogen bond acceptors? The output molecule should be similar to the input molecule.",
    108: "Can you make molecule SMILES_PLACEHOLDER with more hydrogen bond donors? The output molecule should be similar to the input molecule.",

    201: "Can you make molecule SMILES_PLACEHOLDER more soluble in water and more hydrogen bond acceptors? The output molecule should be similar to the input molecule.",
    202: "Can you make molecule SMILES_PLACEHOLDER less soluble in water and more hydrogen bond acceptors? The output molecule should be similar to the input molecule.",
    203: "Can you make molecule SMILES_PLACEHOLDER more soluble in water and more hydrogen bond donors? The output molecule should be similar to the input molecule.",
    204: "Can you make molecule SMILES_PLACEHOLDER less soluble in water and more hydrogen bond donors? The output molecule should be similar to the input molecule.",
    205: "Can you make molecule SMILES_PLACEHOLDER more soluble in water and higher permeability? The output molecule should be similar to the input molecule.",
    206: "Can you make molecule SMILES_PLACEHOLDER more soluble in water and lower permeability? The output molecule should be similar to the input molecule.",
}


def evaluate_result(input_SMILES, output_SMILES, task_id, threshold_list=[0]):
    input_mol = Chem.MolFromSmiles(input_SMILES)
    Chem.Kekulize(input_mol)

    try:
        output_mol = Chem.MolFromSmiles(output_SMILES)
        Chem.Kekulize(output_mol)
    except:
        # print("Invalid output SMILES: {}".format(output_SMILES))
        return None, None, False

    if output_mol is None:
        # print("Invalid output SMILES: {}".format(output_SMILES))
        return None, None, False

    elif task_id == 101:
        prop = "MolLogP"
        threshold = threshold_list[0]
        input_value = prop2func[prop](input_mol)
        output_value = prop2func[prop](output_mol)
        return input_value, output_value, output_value  + threshold < input_value
    
    elif task_id == 102:
        prop = "MolLogP"
        threshold = threshold_list[0]
        input_value = prop2func[prop](input_mol)
        output_value = prop2func[prop](output_mol)
        return input_value, output_value, output_value > input_value + threshold

    elif task_id == 103:
        prop = "qed"
        threshold = threshold_list[0]
        input_value = prop2func[prop](input_mol)
        output_value = prop2func[prop](output_mol)
        return input_value, output_value, output_value > input_value + threshold
    
    elif task_id == 104:
        prop = "qed"
        threshold = threshold_list[0]
        input_value = prop2func[prop](input_mol)
        output_value = prop2func[prop](output_mol)
        return input_value, output_value, output_value + threshold < input_value

    elif task_id == 105:
        prop = "TPSA"
        threshold = threshold_list[0]
        input_value = prop2func[prop](input_mol)
        output_value = prop2func[prop](output_mol)
        return input_value, output_value, output_value + threshold < input_value
    
    elif task_id == 106:
        prop = "TPSA"
        threshold = threshold_list[0]
        input_value = prop2func[prop](input_mol)
        output_value = prop2func[prop](output_mol)
        return input_value, output_value, output_value > input_value + threshold

    elif task_id == 107:
        prop = "NumHAcceptors"
        threshold = threshold_list[0]
        input_value = prop2func[prop](input_mol)
        output_value = prop2func[prop](output_mol)
        return input_value, output_value, output_value > input_value + threshold

    elif task_id == 108:
        prop = "NumHDonors"
        threshold = threshold_list[0]
        input_value = prop2func[prop](input_mol)
        output_value = prop2func[prop](output_mol)
        return input_value, output_value, output_value > input_value + threshold

    elif task_id == 201:
        input_value_01, output_value_01, result_01 = evaluate_result(input_SMILES, output_SMILES, 101, [threshold_list[0]])
        input_value_02, output_value_02, result_02 = evaluate_result(input_SMILES, output_SMILES, 107, [threshold_list[1]])
        return (input_value_01, input_value_02), (output_value_01, output_value_02), result_01 and result_02

    elif task_id == 202:
        input_value_01, output_value_01, result_01 = evaluate_result(input_SMILES, output_SMILES, 102, [threshold_list[0]])
        input_value_02, output_value_02, result_02 = evaluate_result(input_SMILES, output_SMILES, 107, [threshold_list[1]])
        return (input_value_01, input_value_02), (output_value_01, output_value_02), result_01 and result_02

    elif task_id == 203:
        input_value_01, output_value_01, result_01 = evaluate_result(input_SMILES, output_SMILES, 101, [threshold_list[0]])
        input_value_02, output_value_02, result_02 = evaluate_result(input_SMILES, output_SMILES, 108, [threshold_list[1]])
        return (input_value_01, input_value_02), (output_value_01, output_value_02), result_01 and result_02

    elif task_id == 204:
        input_value_01, output_value_01, result_01 = evaluate_result(input_SMILES, output_SMILES, 102, [threshold_list[0]])
        input_value_02, output_value_02, result_02 = evaluate_result(input_SMILES, output_SMILES, 108, [threshold_list[1]])
        return (input_value_01, input_value_02), (output_value_01, output_value_02), result_01 and result_02

    elif task_id == 205:
        input_value_01, output_value_01, result_01 = evaluate_result(input_SMILES, output_SMILES, 101, [threshold_list[0]])
        input_value_02, output_value_02, result_02 = evaluate_result(input_SMILES, output_SMILES, 105, [threshold_list[1]])
        return (input_value_01, input_value_02), (output_value_01, output_value_02), result_01 and result_02

    elif task_id == 206:
        input_value_01, output_value_01, result_01 = evaluate_result(input_SMILES, output_SMILES, 101, [threshold_list[0]])
        input_value_02, output_value_02, result_02 = evaluate_result(input_SMILES, output_SMILES, 106, [threshold_list[1]])
        return (input_value_01, input_value_02), (output_value_01, output_value_02), result_01 and result_02
