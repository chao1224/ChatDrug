
# Dataset and Evaluation Preparation for ChatDrug

First please make and go to the `data` folder:
```
mkdir -p data
cd data
```

And then do the following for dataset and evaluation preparation.

## Small Molecule Editing

- For small molecule editing dataset, please check `small_molecule_editing.txt`. Credit to [MoleculeSTM paper](https://arxiv.org/abs/2212.10789).
- For the retrieval database, please use the ZINC250K dataset from [here](https://github.com/aspuru-guzik-group/chemical_vae/blob/main/models/zinc/250k_rndm_zinc_drugs_clean_3.csv).

## Peptide Editing

- Both the editing and retrieval dataset can be found in [this repo](https://github.com/minrq/pMHC).
- We provide most of the pretrained datasets in `peptide`. You only need to download the `Data_S3.csv` from [this link](https://github.com/minrq/pMHC/blob/main/data/mhcflurry/Data_S3.csv).
- If you want to do the data preprocessing yourself, please refer to the following:
```
cd peptide_editing
python preprocess_step_1_data_extraction.py
python preprocess_step_2_single_prop.py
python preprocess_step_3_multi_prop.py
```

## Protein Editing

- Download dataset from [this google drive](https://drive.google.com/file/d/11szX_dd8NdHfKG5zaMNSDMwP6Vk6EX90/view?usp=share_link).
- Unzip to `protein` folder.
- This includes both the editing and retrieval dataset.
- For evaluation, please download `pytorch_model_ss3.bin` from [this link](https://huggingface.co/chao1224/ProteinCLAP_pretrain_EBM_NCE_downstream_property_prediction). Credit to [ProteinDT](https://arxiv.org/abs/2302.04611).

```
.
├── peptide_editing
│   ├── class1_pseudosequences.csv
│   ├── Data_S3.csv
│   ├── models_class1_presentation
│   │   ├── 10755300.stderr
│   │   .
│   │   .
│   │   .
│   │   └── train_data.csv.bz2
│   ├── peptide_editing.json
│   ├── peptide_editing.json
│   ├── peptide_editing_threshold.json
│   ├── preprocess_step_1_data_extraction.py
│   ├── preprocess_step_2_single_prop.py
│   ├── preprocess_step_3_multi_prop.py
│   └── selected_alleles.txt
├── protein_editing
│   ├── downstream_datasets
│   │   └── secondary_structure
│   │       ├── secondary_structure_casp12.lmdb
│   │       │   ├── data.mdb
│   │       │   └── lock.mdb
│   │       ├── secondary_structure_cb513.lmdb
│   │       │   ├── data.mdb
│   │       │   └── lock.mdb
│   │       ├── secondary_structure_train.lmdb
│   │       │   ├── data.mdb
│   │       │   └── lock.mdb
│   │       ├── secondary_structure_ts115.lmdb
│   │       │   ├── data.mdb
│   │       │   └── lock.mdb
│   │       └── secondary_structure_valid.lmdb
│   │           ├── data.mdb
│   │           └── lock.mdb
│   ├── pytorch_model_ss3.bin
│   └── pytorch_model_ss8.bin
├── README.md
└── small_molecule
    ├── 250k_rndm_zinc_drugs_clean_3.csv
    └── small_molecule_editing.txt
```
