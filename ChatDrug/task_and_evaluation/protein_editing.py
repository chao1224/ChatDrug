import lmdb
import pickle as pkl
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import re
from transformers import BertTokenizerFast


def load_ProteinDT_model(input_model_path, chache_dir, mean_output, num_labels):
    from ChatDrug.TAPE_benchmark.models import BertForTokenClassification2

    model = BertForTokenClassification2.from_pretrained(
        "Rostlab/prot_bert_bfd",
        cache_dir=chache_dir,
        mean_output=mean_output,
        num_labels=num_labels,
    )

    # load model from checkpoint
    print("Loading protein model from {}...".format(input_model_path))
    state_dict = torch.load(input_model_path, map_location='cpu')
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    print("missing keys: {}".format(missing_keys))
    print("unexpected keys: {}".format(unexpected_keys))
    
    return model


# load protein model
device = "cuda"
chache_dir = "data/protein_editing/temp_pretrained_ProteinDT"
input_model_path = "data/protein_editing/pytorch_model_ss3.bin"
protein_model = load_ProteinDT_model(input_model_path, chache_dir, mean_output=True, num_labels=3)
protein_model = protein_model.to(device)
protein_tokenizer = BertTokenizerFast.from_pretrained("Rostlab/prot_bert_bfd", chache_dir=chache_dir, do_lower_case=False)


task_specification_dict_protein = {
    501: "We have a protein PROTEIN_SEQUENCE_PLACEHOLDER. Can you update modify it by making more amino acids into the helix structure (secondary structure)? The input and output protein sequences should be similar but different.",
    502: "We have a protein PROTEIN_SEQUENCE_PLACEHOLDER. Can you update modify it by making more amino acids into the strand structure (secondary structure)? The input and output protein sequences should be similar but different.",
}


def parse_protein(input_protein, raw_text, retrieval_sequence):
    pattern = re.compile('[A-Z]{5,}')
    output_protein_list = pattern.findall(raw_text)
    while input_protein in output_protein_list:
        output_protein_list.remove(input_protein)

    if retrieval_sequence!=None:
        while retrieval_sequence in output_protein_list:
            output_protein_list.remove(retrieval_sequence)

    if len(output_protein_list) > 0:
        output_protein = output_protein_list[0][:1024]
        return [output_protein]
    else:
        return []


def pad_sequences(sequences, constant_value=0, dtype=None) -> np.ndarray:
    batch_size = len(sequences)
    shape = [batch_size] + np.max([seq.shape for seq in sequences], 0).tolist()

    if dtype is None:
        dtype = sequences[0].dtype

    if isinstance(sequences[0], np.ndarray):
        array = np.full(shape, constant_value, dtype=dtype)
    elif isinstance(sequences[0], torch.Tensor):
        array = torch.full(shape, constant_value, dtype=dtype)

    for arr, seq in zip(array, sequences):
        arrslice = tuple(slice(dim) for dim in seq.shape)
        arr[arrslice] = seq

    return array


class ProteinSecondaryStructureDataset(Dataset):
    def __init__(self, data_path, tokenizer, target='ss3'):
        self.tokenizer = tokenizer
        self.target = target
        self.ignore_index = -100

        env = lmdb.open(data_path, max_readers=1, readonly=True,
                        lock=False, readahead=False, meminit=False)

        with env.begin(write=False) as txn:
            num_examples = pkl.loads(txn.get(b'num_examples'))
        
        self.protein_sequence_list = []
        self.ss3_labels_list = []
        self.ss8_labels_list = []

        for index in range(num_examples):
            with env.begin(write=False) as txn:
                item = pkl.loads(txn.get(str(index).encode()))
            # print(item.keys())
            protein_sequence = item["primary"]
            ss3_labels = item["ss3"]
            ss8_labels = item["ss8"]
            protein_length = item["protein_length"]

            if len(protein_sequence) > 1024:
                protein_sequence = protein_sequence[:1024]
                ss3_labels = ss3_labels[:1024]
                ss8_labels = ss8_labels[:1024]
                
            self.protein_sequence_list.append(protein_sequence)
            self.ss3_labels_list.append(ss3_labels)
            self.ss8_labels_list.append(ss8_labels)
        
        if self.target == "ss3":
            self.labels_list = self.ss3_labels_list
            self.num_labels = 3
        else:
            self.labels_list = self.ss8_labels_list
            self.num_labels = 8
        return

    def __len__(self):
        return len(self.labels_list)

    def __getitem__(self, index: int):
        protein_sequence = self.protein_sequence_list[index]
        labels = self.labels_list[index]

        token_ids = self.tokenizer(list(protein_sequence), is_split_into_words=True, return_offsets_mapping=True, truncation=False, padding=True)
        token_ids = np.array(token_ids['input_ids'])
        input_mask = np.ones_like(token_ids)
        
        # pad with -1s because of cls/sep tokens
        labels = np.asarray(labels, np.int64)
        labels = np.pad(labels, (1, 1), 'constant', constant_values=self.ignore_index)

        return token_ids, input_mask, labels

    def collate_fn(self, batch):
        input_ids, input_mask, ss_label = tuple(zip(*batch))
        input_ids = torch.from_numpy(pad_sequences(input_ids, constant_value=self.tokenizer.pad_token_id))
        attention_mask = torch.from_numpy(pad_sequences(input_mask, constant_value=0))
        labels = torch.from_numpy(pad_sequences(ss_label, constant_value=self.ignore_index))

        output = {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}
        return output


def tokenize_one_sequence(tokenizer, protein_sequence):
    token_ids = tokenizer(list(protein_sequence), is_split_into_words=True, return_offsets_mapping=True, truncation=False, padding=True)
    token_ids = np.array(token_ids['input_ids'])
    input_mask = np.ones_like(token_ids)
    return token_ids, input_mask


def tokenize_sequences(tokenizer, sequence_list, labels):
    ignore_index = -100

    input_sequence, output_sequence = sequence_list
    input_token_ids, input_attention_mask = tokenize_one_sequence(tokenizer, input_sequence)
    output_token_ids, output_attention_mask = tokenize_one_sequence(tokenizer, output_sequence)
    token_ids = [input_token_ids, output_token_ids]
    attention_mask = [input_attention_mask, output_attention_mask]

    labels = np.asarray(labels, np.int64)
    labels = np.pad(labels, (1, 1), 'constant', constant_values=ignore_index)
    labels = [labels, labels]  # just duplicate the labels

    token_ids = torch.from_numpy(pad_sequences(token_ids, constant_value=tokenizer.pad_token_id))
    attention_mask = torch.from_numpy(pad_sequences(attention_mask, constant_value=0))
    labels = torch.from_numpy(pad_sequences(labels, constant_value=ignore_index))

    return token_ids, attention_mask, labels


@torch.no_grad()
def evaluate_result(input_protein_sequence, output_protein_sequence, labels, task_id, device="cuda"):
    """
    sequence_list = [input_sequence, output_sequence]
    labels: ground-truth SS-3/SS-8 labels for input_sequence
    """
    sequence_list = [input_protein_sequence, output_protein_sequence]
    input_ids, attention_mask, labels = tokenize_sequences(protein_tokenizer, sequence_list, labels)

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    labels = labels.to(device)

    output = protein_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
    logits = output.logits  # [2, seq_length, 3]
    predicted_labels = F.softmax(logits, dim=-1)  # [2, seq_length, 3]
    predicted_labels = predicted_labels.argmax(dim=-1)  # [2, seq_length]

    if task_id == 501:
        target_label = 0
    elif task_id == 502:
        target_label = 1

    input_predicted_labels, output_predicted_labels = predicted_labels
    input_attention_mask, output_attention_mask = attention_mask
    input_count = ((input_predicted_labels == target_label) * input_attention_mask).sum()
    output_count = ((output_predicted_labels == target_label) * output_attention_mask).sum()

    return input_count, output_count, output_count > input_count


class ProteinListDataset(Dataset):
    def __init__(self, protein_sequence_list, tokenizer, task_id):
        self.tokenizer = tokenizer
        self.ignore_index = -100
        self.protein_sequence_list = protein_sequence_list
        return

    def __len__(self):
        return len(self.protein_sequence_list)

    def __getitem__(self, index: int):
        protein_sequence = self.protein_sequence_list[index]

        token_ids = self.tokenizer(list(protein_sequence), is_split_into_words=True, return_offsets_mapping=True, truncation=False, padding=True)
        token_ids = np.array(token_ids['input_ids'])
        input_mask = np.ones_like(token_ids)

        return token_ids, input_mask

    def collate_fn(self, batch):
        input_ids, input_mask = tuple(zip(*batch))
        input_ids = torch.from_numpy(pad_sequences(input_ids, constant_value=self.tokenizer.pad_token_id))
        attention_mask = torch.from_numpy(pad_sequences(input_mask, constant_value=0))

        output = {'input_ids': input_ids, 'attention_mask': attention_mask}
        return output


@torch.no_grad()
def evaluate_pairwise_list_result(input_protein_list, output_protein_list, task_id, device="cuda"):
    from torch.utils.data import DataLoader

    batch_size = 16
    input_dataset = ProteinListDataset(input_protein_list, tokenizer=protein_tokenizer, task_id=task_id)
    input_dataloader = DataLoader(input_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=input_dataset.collate_fn)

    output_dataset = ProteinListDataset(output_protein_list, tokenizer=protein_tokenizer, task_id=task_id)
    output_dataloader = DataLoader(output_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=input_dataset.collate_fn)

    if task_id == 501:
        target_label = 0
    elif task_id == 502:
        target_label = 1

    def get_target_label_count_list(dataloader, target_label):
        count_list = []
        for batch in dataloader:
            input_ids, attention_mask = batch["input_ids"], batch["attention_mask"]
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            output = protein_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)

            logits = output.logits  # [B, seq_length, 3]
            predicted_labels = F.softmax(logits, dim=-1)  # [B, seq_length, 3]
            predicted_labels = predicted_labels.argmax(dim=-1)  # [B, seq_length]

            temp_count_list = ((predicted_labels == target_label) * attention_mask)
            temp_count_list = temp_count_list.sum(dim=1)  # [B]
            count_list.append(temp_count_list.detach().cpu().numpy())
        
        count_list = np.concatenate(count_list)
        print("count_list", count_list.shape)
        return count_list

    input_count_list = get_target_label_count_list(input_dataloader, target_label)
    output_count_list = get_target_label_count_list(output_dataloader, target_label)

    return input_count_list, output_count_list, output_count_list > input_count_list


@torch.no_grad()
def evaluate_fast_protein_dict(input_protein_list, task_id, device="cuda"):
    from torch.utils.data import DataLoader

    batch_size = 128
    input_dataset = ProteinListDataset(input_protein_list, tokenizer=protein_tokenizer, task_id=task_id)
    input_dataloader = DataLoader(input_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=input_dataset.collate_fn)

    if task_id == 501:
        target_label = 0
    elif task_id == 502:
        target_label = 1

    def get_target_label_count_list(dataloader, target_label):
        count_list = []
        for batch in dataloader:
            input_ids, attention_mask = batch["input_ids"], batch["attention_mask"]
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            output = protein_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)

            logits = output.logits  # [B, seq_length, 3]
            predicted_labels = F.softmax(logits, dim=-1)  # [B, seq_length, 3]
            predicted_labels = predicted_labels.argmax(dim=-1)  # [B, seq_length]

            temp_count_list = ((predicted_labels == target_label) * attention_mask)
            temp_count_list = temp_count_list.sum(dim=1)  # [B]
            count_list.append(temp_count_list.detach().cpu().numpy())
        
        count_list = np.concatenate(count_list)
        print("count_list", count_list.shape)
        return count_list

    input_count_list = get_target_label_count_list(input_dataloader, target_label)

    return input_count_list


@torch.no_grad()
def evaluate_fast_protein(input_protein_list, output_protein_list, task_id, dict_sequence, device="cuda"):
    from torch.utils.data import DataLoader

    batch_size = 1
    output_dataset = ProteinListDataset(output_protein_list, tokenizer=protein_tokenizer, task_id=task_id)
    output_dataloader = DataLoader(output_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=output_dataset.collate_fn)

    if task_id == 501:
        target_label = 0
    elif task_id == 502:
        target_label = 1

    def get_target_label_count_list(dataloader, target_label):
        count_list = []
        for batch in dataloader:
            input_ids, attention_mask = batch["input_ids"], batch["attention_mask"]
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            output = protein_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)

            logits = output.logits  # [B, seq_length, 3]
            predicted_labels = F.softmax(logits, dim=-1)  # [B, seq_length, 3]
            predicted_labels = predicted_labels.argmax(dim=-1)  # [B, seq_length]

            temp_count_list = ((predicted_labels == target_label) * attention_mask)
            temp_count_list = temp_count_list.sum(dim=1)  # [B]
            count_list.append(temp_count_list.detach().cpu().numpy())
        
        count_list = np.concatenate(count_list)
        print("count_list", count_list.shape)
        return count_list

    output_count_list = get_target_label_count_list(output_dataloader, target_label)

    input_count_list = []
    for sequence in input_protein_list:
        input_count_list.append(dict_sequence[sequence])

    return output_count_list > input_count_list