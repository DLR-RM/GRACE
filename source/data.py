import torch
import pytorch_lightning as pl
import os
import math

from torch_geometric.loader import DataLoader
from torch_geometric.data import InMemoryDataset
import torch_geometric.transforms as T

from source.transform import *
from source.dataset import remove_files
from source.utils import ImbalancedPartsDatasetSampler


class AssemblyFoldDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['raw.pt']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        data_list = torch.load(self.raw_paths[0])

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class AssemblyDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return os.listdir(self.raw_dir)

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        data_list = [torch.load(f) for f in self.raw_paths]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class GraphDataModuleBase(pl.LightningDataModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(dict(hparams))

    def split_dataset(self, train_ratio=0.9, val_ratio=0.1):
        dataset = self.dataset.shuffle()

        nof_samples = len(dataset)
        train_size = math.floor(train_ratio * nof_samples)
        val_size = math.floor(val_ratio * nof_samples)

        train_set = dataset.index_select(slice(0, train_size))
        val_set = dataset.index_select(slice(train_size, train_size + val_size))

        return train_set, val_set

    def _init_pre_transform(self):
        trans_list = list()
        if self.hparams.pos_encoding:
            trans_list.append(NodePositionalEncoding(
                position_index=[('part', 2), ('surface', 1)],
                encoding_length=self.hparams.encoding_length
            ))
        if self.hparams.one_hot:
            # for part/surface types
            trans_list.append(NodeOneHot(
                position_index=[('part', 1), ('surface', 0)]
            ))
        if self.hparams.one_hot_positional:
            trans_list.append(NodeOneHot(
                position_index=[('surface', 1)],
                one_hot_max_val=32
            ))
            trans_list.append(NodeOneHot(
                position_index=[('part', 2)],
                one_hot_max_val=32
            ))

        if self.hparams.edge_normalization:
            trans_list.append(EdgeNormalizationByParts())
        
        if self.hparams.remove_pos_encodings:
            trans_list.append(RemovePositionalEncoding(
                position_index=[('part', 2), ('surface', 1)],
            ))

        return T.Compose(trans_list)
    
    def _init_transform(self):
        trans_list = list()
        if self.hparams.permute_nodes:
            trans_list.append(PermuteNodeOrder(
                position_index=[('part', 2), ('surface', 1)],
                encoding_length=self.hparams.encoding_length
            ))
        
        if self.hparams.permute_parts:
            trans_list.append(PermuteNodeOrder(
                position_index=[('part', 2)],
                encoding_length=self.hparams.encoding_length
            ))

        if self.hparams.permute_surfaces:
            trans_list.append(PermuteNodeOrder(
                position_index=[('surface', 1)],
                encoding_length=self.hparams.encoding_length
            ))

        if self.hparams.random_pos_encoding:
            trans_list.append(RandomPositionalEncoding(
                position_index=[('part', 2), ('surface', 1)],
                encoding_length=self.hparams.encoding_length
            ))

        return T.Compose(trans_list)


class  GraphFoldDataModule(GraphDataModuleBase):
    def __init__(self, hparams, dataset, mode="trainval", train_index=None, val_index=None, balanced_train_sampler=False):
        super().__init__(hparams)

        self.pre_transform = self._init_pre_transform()
        self.transform = self._init_transform()
        self.construct_mode = mode
        self.balanced_train_sampler = balanced_train_sampler
        self._setup(dataset, train_index, val_index)

    def _setup(self, dataset, train_index, val_index):
        cache_root = self.hparams.dataset_path

        # huge hack of InMemory dataloader !
        self._reset_dataloader_cache(cache_root, dataset)
        self.dataset = AssemblyFoldDataset(root=cache_root,
                                           transform=None,
                                           pre_transform=self.pre_transform)

        print(f'Graphs loaded: {len(self.dataset)}')
        
        if self.construct_mode == "trainval":        
            if len(val_index):
                self.train_set = self.dataset.index_select(train_index)
                self.val_set = self.dataset.index_select(val_index)
            else:
                self.train_set, self.val_set = self.split_dataset(self.hparams.train_ratio, 1 - self.hparams.train_ratio)

            # we want to set transformations only for train_set!
            self.train_set.transform = self.transform

            print(f'  Train: {len(self.train_set)}, Val: {len(self.val_set)}')
        else:
            self.test_set = self.dataset
            print(f'  Test: {len(self.test_set)}')

    def _reset_dataloader_cache(self, cache_root, dataset):
        # remove previous cache
        raw_dir_path = os.path.join(cache_root, "raw/")
        remove_files(raw_dir_path)
        
        processed_dir_path = os.path.join(cache_root, "processed/")
        remove_files(processed_dir_path)

        # create cache file used by dataset
        os.makedirs(os.path.join(cache_root, "raw/"), exist_ok=True)
        torch.save(dataset, os.path.join(cache_root, "raw/raw.pt"))

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_set, 
            batch_size=self.hparams.batch_size, 
            shuffle=True if not self.balanced_train_sampler else False, 
            num_workers=self.hparams.num_workers, 
            pin_memory=True,
            sampler=ImbalancedPartsDatasetSampler(self.train_set) if self.balanced_train_sampler else None
        )

    def val_dataloader(self, batch_size=None):
        return DataLoader(self.val_set, 
                          batch_size=self.hparams.batch_size if batch_size is None else batch_size, 
                          num_workers=self.hparams.num_workers,
                          pin_memory=True)

    def predict_dataloader(self, indices=None, batch_size=1):
        self.predict_set = self.test_set
            
        return DataLoader(self.predict_set, batch_size=batch_size, num_workers=1, shuffle=False)


class  GraphDataModule(GraphDataModuleBase):
    def __init__(self, hparams):
        super().__init__(hparams)

        self.pre_transforms = self._init_pre_transform()
        self.transforms = self._init_transform()
        self._setup()

    def _setup(self):
        print("Loading dataset: ", self.hparams.dataset_path)
        self.dataset = AssemblyDataset(root=self.hparams.dataset_path,
                                       transform=self.transforms,
                                       pre_transform=self.pre_transforms
        )

        print(f'Graphs loaded: {len(self.dataset)}')

        train_ratio = self.hparams.train_ratio
        val_ratio = 1 - train_ratio
        self.train_set, self.val_set = self.split_dataset(train_ratio, val_ratio)

        print(f'Train: {len(self.train_set)}, Val: {len(self.val_set)}')

    def train_dataloader(self):
        return DataLoader(self.train_set, 
                          batch_size=self.hparams.batch_size, 
                          shuffle=False, # TODO - FIXME!!!!!!
                          num_workers=self.hparams.num_workers, 
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, 
                          batch_size=self.hparams.batch_size, 
                          num_workers=self.hparams.num_workers, 
                          pin_memory=True)


    def predict_dataloader(self, val_set=False, indices=None, batch_size=256):
        if val_set:
            self.predict_set = self.val_set
            print("Using val set")
        else:
            self.predict_set = self.dataset

        if indices:    
            self.predict_set = torch.utils.data.Subset(self.predict_set, indices)
            print("Using specific provided indices")
            
        return DataLoader(self.predict_set, batch_size=batch_size, num_workers=self.hparams.num_workers, shuffle=False)
