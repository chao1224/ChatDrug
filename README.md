# ChatGPT-powered Conversational Drug Editing Using Retrieval and Domain Feedback

Authors: Shengchao Liu<sup>+</sup>, Jiongxiao Wang<sup>+</sup>, Yijin Yang, Chengpeng Wang, Ling Liu, Hongyu Guo<sup>\*</sup>, Chaowei Xiao<sup>\*</sup>

<sup>+</sup> Equal contribution<br>
<sup>\*</sup> Equal advising

[[Project Page](https://chao1224.github.io/ChatDrug)]
[[ArXiv](https://arxiv.org/abs/2305.18090)]

<p align="center">
  <img src="figure/pipeline.png" /> 
</p>


ChatDrug is for conversational drug editing, and three types of drugs are considered:
- Small Molecules
- Peptides
- Proteins
<p align="left">
  <img src="figure/final_demo.gif" width="100%" /> 
</p>

## Environment

Setup the anaconda (skip this if you already have conda)
 ```bash
wget https://repo.continuum.io/archive/Anaconda3-2019.10-Linux-x86_64.sh
bash Anaconda3-2019.10-Linux-x86_64.sh -b
export PATH=$PWD/anaconda3/bin:$PATH
```

Then download the required python packages:
```bash
conda create -n ChatDrug python=3.7
conda activate ChatDrug
conda install -y -c rdkit rdkit
conda install -y numpy networkx scikit-learn
conda install -y -c conda-forge -c pytorch pytorch=1.9.1

pip install tensorflow
pip install mhcflurry
pip install levenshtein

pip install transformers
pip install lmdb
pip install seqeval
pip install openai

pip install -e .
```

## Dataset

We provide the dataset in [this link](https://huggingface.co/datasets/chao1224/ChatDrug_data). You can manually download and move to the `data` folder or using the following python script.
```
from huggingface_hub import snapshot_download

snapshot_download(repo_id="chao1224/ChatDrug_data", repo_type="dataset", local_dir="data", local_dir_use_symlinks=False, ignore_patterns=["README.md"])
```
Please give credits to the original papers. For more details of dataset, please check the [data folder](./data).

## Evaluation

The evaluation metrics for three editing tasks are below:
| Drug Type | Evaluation |
| -- | -- |
| Small Molecule | RDKit (`conda install -y -c rdkit rdkit`)|
| Peptide | [MHCFlurry](https://github.com/openvax/mhcflurry)|
| Protein | [ProteinDT paper](https://arxiv.org/abs/2302.04611), [checkpoints](https://huggingface.co/chao1224/ProteinCLAP_pretrain_EBM_NCE_downstream_property_prediction) |

For evaluation on peptides and proteins, please read the following instructions:
- For peptides (MHCFlurry), please run the following bash commands:
```
> pip install mhcflurry
> mhcflurry-downloads fetch models_class1_presentation
> mhcflurry-downloads path models_class1_presentation
$PATH
> mv $PATH data/peptide/models_class1_presentation
```
- For proteins (ProteinDT / ProteinCLAP), please run the following python script:
```
from huggingface_hub import hf_hub_download

hf_hub_download(
  repo_id="chao1224/ProteinCLAP_pretrain_EBM_NCE_downstream_property_prediction",
  repo_type="model",
  filename="pytorch_model_ss3.bin",
  cache_dir="data/protein")
```
Please give credits to the original papers. For more details of evaluation, please check the [data folder](./data).

## Prompt for Drug Editing

All the task prompts are defined in `ChatDrug/task_and_evaluation`. you can also find it on [the hugging face link](https://huggingface.co/datasets/chao1224/ChatDrug_prompt).

## Usage

Please provide your OpenAI API in `model_utils.py`

To use ChatDrug, you need to implement PPDS module first to obtain the raw answer from ChatGPT:
```
python PPDS.py --tasks task_id --saved_file save_path
```
Then start conversation with ChatGPT by:
```
python Conversation.py --tasks task_id --num_round conversation_round --saved_file save_path
```
Results will be saved in `save_path`.

For protein editing tasks, multiple evaluation times in retrieval process would consume a lot of time. Thus, we provide a fast version of conversation setting, which requires to firstly compute evaluation score of protein sequences in retrieval data base and test dataset. Running the following command to implement accelerate ChatDrug for protein editing tasks:
```
python protein_generate_retrieval_dict.py --tasks 5
python Conversation_protein_fast.py --tasks 5 --num_round conversation_round --saved_file save_path
```

We also provide code for In-Context Learning setting:
```
python In-Context.py --tasks task_id --saved_file save_path
```


## Cite Us
Feel free to cite this work if you find it useful to you!

```
@article{liu2023chatdrug,
    title={ChatGPT-powered Conversational Drug Editing Using Retrieval and Domain Feedback},
    author={Shengchao Liu, Jiongxiao Wang, Yijin Yang, Chengpeng Wang, Ling Liu, Hongyu Guo, Chaowei Xiao},
    journal={arXiv preprint arXiv:2305.18090},
    year={2023}
}
```
