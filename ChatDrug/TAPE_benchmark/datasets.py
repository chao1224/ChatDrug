from pathlib import Path
from typing import Union

import pickle as pkl
import lmdb
import numpy as np
import pandas as pd
import re
import torch
from scipy.spatial.distance import squareform, pdist
from typing import Union, List, Tuple, Sequence, Dict, Any, Optional, Collection
from torch.utils.data import Dataset
import os


def dataset_factory(data_file: Union[str, Path], *args, **kwargs) -> Dataset:
    data_file = Path(data_file)
    if not data_file.exists():
        raise FileNotFoundError(data_file)
    if data_file.suffix == '.lmdb':
        return LMDBDataset(data_file, *args, **kwargs)
    elif data_file.suffix in {'.fasta', '.fna', '.ffn', '.faa', '.frn'}:
        return FastaDataset(data_file, *args, **kwargs)
    elif data_file.suffix == '.json':
        return JSONDataset(data_file, *args, **kwargs)
    elif data_file.is_dir():
        return NPZDataset(data_file, *args, **kwargs)
    else:
        raise ValueError(f"Unrecognized datafile type {data_file.suffix}")


def pad_sequences(sequences: Sequence, constant_value=0, dtype=None) -> np.ndarray:
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


class LMDBDataset(Dataset):
    def __init__(self, data_file, in_memory):
        env = lmdb.open(data_file, max_readers=1, readonly=True,
                        lock=False, readahead=False, meminit=False)

        with env.begin(write=False) as txn:
            num_examples = pkl.loads(txn.get(b'num_examples'))

        if in_memory:
            cache = [None] * num_examples
            self._cache = cache

        self._env = env
        self._in_memory = in_memory
        self._num_examples = num_examples

    def __len__(self):
        return self._num_examples

    def __getitem__(self, index):
        if self._in_memory and self._cache[index] is not None:
            item = self._cache[index]
        else:
            with self._env.begin(write=False) as txn:
                item = pkl.loads(txn.get(str(index).encode()))
                if 'id' not in item:
                    item['id'] = str(index)
                if self._in_memory:
                    self._cache[index] = item
        return item


class DataProcessor:
    """Base class for data converters for biological tasks data sets."""
    def get_train_examples(self, data_dir):
        """Gets a collection of :class:`InputExample` for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of :class:`InputExample` for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of :class:`InputExample` for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()


class FluorescenceProgress(DataProcessor):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer

    def get_train_examples(self, data_dir, in_memory=True):
        dataset = FluorescenceDataset(data_dir, split='train', tokenizer=self.tokenizer)
        return dataset

    def get_dev_examples(self, data_dir, in_memory=True):
        dataset = FluorescenceDataset(data_dir, split='valid', tokenizer=self.tokenizer)
        return dataset

    def get_test_examples(self, data_dir, data_cat, in_memory=True):
        if data_cat is not None:
            dataset = FluorescenceDataset(data_dir, split=data_cat, tokenizer=self.tokenizer)
        else:
            dataset = FluorescenceDataset(data_dir, split='test', tokenizer=self.tokenizer)
        return dataset

    def get_labels(self):
        return list(range(1))


class SecondaryStructureProcessor3(DataProcessor):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer

    def get_train_examples(self, data_dir, in_memory=True):
        dataset = SecondaryStructureDataset3(data_dir, split='train', tokenizer=self.tokenizer, target='ss3', in_memory=in_memory)
        return dataset

    def get_dev_examples(self, data_dir, in_memory=True):
        dataset = SecondaryStructureDataset3(data_dir, split='valid', tokenizer=self.tokenizer, target='ss3', in_memory=in_memory)
        return dataset

    def get_test_examples(self, data_dir, data_cat, in_memory=True):
        dataset = SecondaryStructureDataset3(data_dir, split=data_cat, tokenizer=self.tokenizer, target='ss3', in_memory=in_memory)
        return dataset

    def get_labels(self):
        return list(range(3))


class SecondaryStructureProcessor8(DataProcessor):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer

    def get_train_examples(self, data_dir, in_memory=True):
        dataset = SecondaryStructureDataset8(data_dir, split='train', tokenizer=self.tokenizer, target='ss8', in_memory=in_memory)
        return dataset

    def get_dev_examples(self, data_dir, in_memory=True):
        dataset = SecondaryStructureDataset8(data_dir, split='valid', tokenizer=self.tokenizer, target='ss8', in_memory=in_memory)
        return dataset

    def get_test_examples(self, data_dir, data_cat, in_memory=True):
        dataset = SecondaryStructureDataset8(data_dir, split=data_cat, tokenizer=self.tokenizer, target='ss8', in_memory=in_memory)
        return dataset

    def get_labels(self):
        return list(range(8))


class ContactProgress(DataProcessor):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer

    def get_train_examples(self, data_dir, in_memory=True):
        dataset = ProteinnetDataset(data_dir, split='train', tokenizer=self.tokenizer)
        return dataset

    def get_dev_examples(self, data_dir, in_memory=True):
        dataset = ProteinnetDataset(data_dir, split='valid', tokenizer=self.tokenizer)
        return dataset

    def get_test_examples(self, data_dir, data_cat, in_memory=True):
        if data_cat is not None:
            dataset = ProteinnetDataset(data_dir, split=data_cat, tokenizer=self.tokenizer)
        else:
            dataset = ProteinnetDataset(data_dir, split='test', tokenizer=self.tokenizer)
        return dataset

    def get_labels(self):
        return list(range(2))


class StabilityProgress(DataProcessor):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer

    def get_train_examples(self, data_dir, in_memory=True):
        dataset = StabilityDataset(data_dir, split='train', tokenizer=self.tokenizer)
        return dataset

    def get_dev_examples(self, data_dir, in_memory=True):
        dataset = StabilityDataset(data_dir, split='valid', tokenizer=self.tokenizer)
        return dataset

    def get_test_examples(self, data_dir, data_cat, in_memory=True):
        if data_cat is not None:
            dataset = StabilityDataset(data_dir, split=data_cat, tokenizer=self.tokenizer)
        else:
            dataset = StabilityDataset(data_dir, split='test', tokenizer=self.tokenizer)
        return dataset

    def get_labels(self):
        return list(range(1))


class RemoteHomologyProgress(DataProcessor):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer

    def get_train_examples(self, data_dir, in_memory=True):
        dataset = RemoteHomologyDataset(data_dir, split='train', tokenizer=self.tokenizer)
        return dataset

    def get_dev_examples(self, data_dir, in_memory=True):
        dataset = RemoteHomologyDataset(data_dir, split='valid', tokenizer=self.tokenizer)
        return dataset

    def get_test_examples(self, data_dir, data_cat, in_memory=True):
        if data_cat is not None:
            dataset = RemoteHomologyDataset(data_dir, split=data_cat, tokenizer=self.tokenizer)
        else:
            dataset = RemoteHomologyDataset(data_dir, split='test', tokenizer=self.tokenizer)
        return dataset

    def get_labels(self):
        return list(range(1195))


class ProteinnetDataset(Dataset):

    def __init__(self,
                 data_path: Union[str, Path],
                 split: str,
                 tokenizer):

        if split not in ('train', 'train_unfiltered', 'valid', 'test'):
            raise ValueError(f"Unrecognized split: {split}. Must be one of "
                             f"['train', 'train_unfiltered', 'valid', 'test']")

        self.tokenizer = tokenizer

        data_path = Path(data_path)
        data_file = f'proteinnet/proteinnet_{split}.json'
        self.data = dataset_factory(data_path / data_file)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        item = self.data[index]

        seq = list(re.sub(r"[UZOB]", "X", item['primary']))
        token_ids = self.tokenizer(seq, is_split_into_words=True)
        token_ids = np.asarray(token_ids['input_ids'], dtype=int)
        protein_length = len(seq)
        #if protein_length > 1000:
        #    print(seq)
        input_mask = np.ones_like(token_ids)

        valid_mask = item['valid_mask']
        valid_mask = np.array(valid_mask)
        #print("type:", type(valid_mask))
        #print("valid_mask", valid_mask)
        contact_map = np.less(squareform(pdist(torch.tensor(item['tertiary']))), 8.0).astype(np.int64)

        yind, xind = np.indices(contact_map.shape)
        # DEL
        invalid_mask = ~(valid_mask[:, None] & valid_mask[None, :])
        invalid_mask |= np.abs(yind - xind) < 6
        contact_map[invalid_mask] = -1

        return token_ids, protein_length, input_mask, contact_map

    def collate_fn(self, batch):
        input_ids, protein_length, input_mask, contact_labels = tuple(zip(*batch))

        input_ids = torch.from_numpy(pad_sequences(input_ids, 0))
        input_mask = torch.from_numpy(pad_sequences(input_mask, 0))
        contact_labels = torch.from_numpy(pad_sequences(contact_labels, -1))
        protein_length = torch.LongTensor(protein_length)  # type: ignore

        return {'input_ids': input_ids,
                'attention_mask': input_mask,
                'labels': contact_labels,
                'protein_length': protein_length}


class FluorescenceDataset(Dataset):
    def __init__(self, file_path, split, tokenizer):
        self.tokenizer = tokenizer
        self.file_path = file_path

        if split not in ('train', 'valid', 'test'):
            raise ValueError(f"Unrecognized split: {split}. Must be one of "
                             f"['train', 'valid', 'test'")

        data_file = f'{self.file_path}/fluorescence/fluorescence_{split}.json'
        self.seqs, self.labels = self.get_data(data_file)

    def get_data(self, file):
        # print(file)
        fp = pd.read_json(file)
        seqs = fp.primary
        labels = fp.log_fluorescence

        return seqs, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        seq = list(re.sub(r"[UZOB]", "X", self.seqs[index]))

        input_ids = self.tokenizer(seq, is_split_into_words=True, truncation=True, padding="max_length", max_length=239)
        input_ids = np.array(input_ids['input_ids'])
        input_mask = np.ones_like(input_ids)

        label = self.labels[index]

        return input_ids, input_mask, label

    def collate_fn(self, batch):
        input_ids, input_mask, fluorescence_true_value = tuple(zip(*batch))
        input_ids = torch.from_numpy(pad_sequences(input_ids, 0))
        input_mask = torch.from_numpy(pad_sequences(input_mask, 0))
        fluorescence_true_value = torch.FloatTensor(fluorescence_true_value)  # type: ignore

        #print(fluorescence_true_value.shape)
        return {'input_ids': input_ids,
                'attention_mask': input_mask,
                'labels': fluorescence_true_value}

class StabilityDataset(Dataset):
    def __init__(self, file_path, split, tokenizer):
        self.file_path = file_path
        self.tokenizer = tokenizer

        if split not in ('train', 'valid', 'test'):
            raise ValueError(f"Unrecognized split: {split}. Must be one of "
                             f"['train', 'valid', 'test'")

        data_file = f'{self.file_path}/stability/stability_{split}.json'
        self.seqs, self.labels = self.get_data(data_file)

    def get_data(self, path):
        read_file = pd.read_json(path)

        seqs = read_file.primary
        labels = read_file.stability_score

        return seqs, labels

    def __getitem__(self, index):
        seq = list(re.sub(r"[UZOB]", "X", self.seqs[index]))

        input_ids = self.tokenizer(seq, is_split_into_words=True, padding="max_length", max_length=50, truncation=True)
        input_ids = np.array(input_ids['input_ids'])
        input_mask = np.ones_like(input_ids)

        label = self.labels[index]

        return input_ids, input_mask, label

    def __len__(self):
        return len(self.labels)

    def collate_fn(self, batch):
        input_ids, input_mask, stability_true_value = tuple(zip(*batch))
        input_ids = torch.from_numpy(pad_sequences(input_ids, 0))
        input_mask = torch.from_numpy(pad_sequences(input_mask, 0))
        stability_true_value = torch.FloatTensor(stability_true_value)  # type: ignore

        return {'input_ids': input_ids,
                'attention_mask': input_mask,
                'labels': stability_true_value}


class RemoteHomologyDataset(Dataset):
    def __init__(self, file_path, split, tokenizer):
        self.tokenizer = tokenizer
        self.file_path = file_path

        if split not in ('train', 'valid', 'test_fold_holdout',
                         'test_family_holdout', 'test_superfamily_holdout'):
            raise ValueError(f"Unrecognized split: {split}. Must be one of "
                             f"['train', 'valid', 'test_fold_holdout', "
                             f"'test_family_holdout', 'test_superfamily_holdout']")

        data_file = f'{self.file_path}/remote_homology/remote_homology_{split}.json'

        self.seqs, self.labels = self.get_data(data_file)

    def get_data(self, file):
        fp = pd.read_json(file)

        seqs = fp.primary
        labels = fp.fold_label

        return seqs, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        seq = list(re.sub(r"[UZOB]", "X", self.seqs[index]))

        input_ids = self.tokenizer(seq, is_split_into_words=True, truncation=True, padding="max_length", max_length=512)
        input_ids = np.array(input_ids['input_ids'])
        input_mask = np.ones_like(input_ids)

        label = self.labels[index]

        return input_ids, input_mask, label

    def collate_fn(self, batch):
        input_ids, input_mask, fold_label = tuple(zip(*batch))
        input_ids = torch.from_numpy(pad_sequences(input_ids, 0))
        input_mask = torch.from_numpy(pad_sequences(input_mask, 0))
        fold_label = torch.LongTensor(fold_label)  # type: ignore

        return {'input_ids': input_ids,
                'attention_mask': input_mask,
                'labels': fold_label}


class SecondaryStructureDataset3(Dataset):
    def __init__(
            self,
            data_path,
            split,
            tokenizer,
            in_memory,
            target='ss3'
    ):
        self.tokenizer = tokenizer
        data_file = f'secondary_structure/secondary_structure_{split}.lmdb'
        print(data_file)
        self.data = LMDBDataset(data_file=os.path.join(data_path, data_file), in_memory=in_memory)
        self.target = target
        print("in total {} proteins".format(len(self.data)))

        self.ignore_index: int = -100

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        item = self.data[index]
        if len(item['primary']) > 1024:
            item['primary'] = item['primary'][:1024]
            item['ss3'] = item['ss3'][:1024]
        token_ids = self.tokenizer(list(item['primary']), is_split_into_words=True, return_offsets_mapping=True, truncation=False, padding=True)
        token_ids = np.array(token_ids['input_ids'])
        input_mask = np.ones_like(token_ids)
        
        # pad with -1s because of cls/sep tokens
        labels = np.asarray(item['ss3'], np.int64)
        labels = np.pad(labels, (1, 1), 'constant', constant_values=self.ignore_index)

        return token_ids, input_mask, labels

    def collate_fn(self, batch):
        input_ids, input_mask, ss_label = tuple(zip(*batch))
        input_ids = torch.from_numpy(pad_sequences(input_ids, constant_value=self.tokenizer.pad_token_id))
        attention_mask = torch.from_numpy(pad_sequences(input_mask, constant_value=0))
        labels = torch.from_numpy(pad_sequences(ss_label, constant_value=self.ignore_index))

        output = {'input_ids': input_ids,
                  'attention_mask': attention_mask,
                  'labels': labels}

        return output


class SecondaryStructureDataset8(Dataset):
    def __init__(
            self,
            data_path,
            split,
            tokenizer,
            in_memory,
            target='ss8'
    ):
        self.tokenizer = tokenizer
        data_file = f'secondary_structure/secondary_structure_{split}.lmdb'
        self.data = LMDBDataset(data_file=os.path.join(data_path, data_file), in_memory=in_memory)
        self.target = target

        self.ignore_index: int = -100

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        item = self.data[index]
        if len(item['primary']) > 1024:
            item['primary'] = item['primary'][:1024]
            item['ss8'] = item['ss8'][:1024]
        token_ids = self.tokenizer(list(item['primary']), is_split_into_words=True, return_offsets_mapping=True, truncation=False, padding=True)
        token_ids = np.array(token_ids['input_ids'])
        input_mask = np.ones_like(token_ids)

        # pad with -1s because of cls/sep tokens
        labels = np.asarray(item['ss8'], np.int64)
        labels = np.pad(labels, (1, 1), 'constant', constant_values=self.ignore_index)

        return token_ids, input_mask, labels

    def collate_fn(self, batch):
        input_ids, input_mask, ss_label = tuple(zip(*batch))

        input_ids = torch.from_numpy(pad_sequences(input_ids, constant_value=self.tokenizer.pad_token_id))
        attention_mask = torch.from_numpy(pad_sequences(input_mask, constant_value=0))
        labels = torch.from_numpy(pad_sequences(ss_label, constant_value=self.ignore_index))

        output = {'input_ids': input_ids,
                  'attention_mask': attention_mask,
                  'labels': labels}

        return output


output_mode_mapping = {
    'ss3': 'token-level-classification',
    'ss8': 'token-level-classification',
    'contact': 'token-level-classification',
    'remote_homology': 'sequence-level-classification',
    'fluorescence': 'sequence-level-regression',
    'stability': 'sequence-level-regression',
}

dataset_processor_mapping = {
    'remote_homology': RemoteHomologyProgress,
    'fluorescence': FluorescenceProgress,
    'stability': StabilityProgress,
    'contact': ContactProgress,
    'ss3': SecondaryStructureProcessor3,
    'ss8': SecondaryStructureProcessor8
}


if __name__ == "__main__":
    from transformers import BertTokenizer
    protein_tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False, chache_dir="../../data/temp_pretrained_PotBert")
    data_dir = "../../data/downstream_datasets"

    dataset_name_list = ['ss3', 'ss8', 'contact', 'remote_homology', 'fluorescence', 'stability']

    for dataset_name in dataset_name_list:
        print(dataset_name)
        output_mode = output_mode_mapping[dataset_name]
        processor = dataset_processor_mapping[dataset_name](protein_tokenizer)
        num_labels = len(processor.get_labels())
        print("num labels: {}".format(num_labels))

        train_dataset = (processor.get_train_examples(data_dir=data_dir))
        eval_dataset = (processor.get_dev_examples(data_dir=data_dir))
        print("train_dataset", len(train_dataset))
        print("eval_dataset", len(eval_dataset))

        if dataset_name == 'remote_homology':
            test_fold_dataset = (
                processor.get_test_examples(data_dir=data_dir, data_cat='test_fold_holdout')
            )
            test_family_dataset = (
                processor.get_test_examples(data_dir=data_dir, data_cat='test_family_holdout')
            )
            test_superfamily_dataset = (
                processor.get_test_examples(data_dir=data_dir, data_cat='test_superfamily_holdout')
            )
            print("test_fold_dataset", len(test_fold_dataset))
            print("test_family_dataset", len(test_family_dataset))
            print("test_superfamily_dataset", len(test_superfamily_dataset))
            print("test in total", len(test_fold_dataset) + len(test_family_dataset) + len(test_superfamily_dataset))

        elif dataset_name == 'ss3' or dataset_name == 'ss8':
            cb513_dataset = (
                processor.get_test_examples(data_dir=data_dir, data_cat='cb513')
            )
            ts115_dataset = (
                processor.get_test_examples(data_dir=data_dir, data_cat='ts115')
            )
            casp12_dataset = (
                processor.get_test_examples(data_dir=data_dir, data_cat='casp12')
            )
            print("cb513_dataset", len(cb513_dataset))
            print("ts115_dataset", len(ts115_dataset))
            print("casp12_dataset", len(casp12_dataset))
            print("test in total", len(cb513_dataset) + len(ts115_dataset) + len(casp12_dataset))
            
        else:
            test_dataset = (
                processor.get_test_examples(data_dir=data_dir, data_cat='test')
            )
            print("test_dataset", len(test_dataset))
        print()
