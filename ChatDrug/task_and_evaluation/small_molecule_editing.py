from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

props = ["MolLogP", "qed", "TPSA", "NumHAcceptors", "NumHDonors"]
prop_pred = [(n, func) for n, func in Descriptors.descList if n.split("_")[-1] in props]

prop2func = {}
for prop, func in prop_pred:
    prop2func[prop] = func


task_specification_dict_molecule = {
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


task2threshold_list = {
    101: [[0], [0.5]],
    102: [[0], [0.5]],
    103: [[0], [0.1]],
    104: [[0], [0.1]],
    105: [[0], [10]],
    106: [[0], [10]],
    107: [[0], [1]],
    108: [[0], [1]],

    201: [[0, 0], [0.5, 1]],
    202: [[0, 0], [0.5, 1]],
    203: [[0, 0], [0.5, 1]],
    204: [[0, 0], [0.5, 1]],
    205: [[0, 0], [0.5, 10]],
    206: [[0, 0], [0.5, 10]],
}


def parse_molecule(input_sequence, raw_text, retrieval_sequence):
    record=[]
    for line in raw_text.strip().split("\n"):
        line = line.strip()
        # for same molecule return
        line += " "
        line = line.replace(input_sequence + " ", " ")
        if retrieval_sequence != None:
            line = line.replace(retrieval_sequence + " ", " ")
        if line.startswith("- "):
            output_SMILES = line[2:]
            output_SMILES = output_SMILES.split(" ")[0]
            record.append(output_SMILES)
    while '' in record:
        record.remove('')

    return record


def evaluate_molecule(input_SMILES, output_SMILES, task_id, threshold_list=[0]):
>>>>>>> cbd39786dacc7cc624594191c2e36b19f9f1cc6a
    input_mol = Chem.MolFromSmiles(input_SMILES)
    Chem.Kekulize(input_mol)

    try:
        output_mol = Chem.MolFromSmiles(output_SMILES)
        Chem.Kekulize(output_mol)
    except:
        # print("Invalid output SMILES: {}".format(output_SMILES))
        return None, None, -1

    if output_mol is None:
        # print("Invalid output SMILES: {}".format(output_SMILES))
        return None, None, -1

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
