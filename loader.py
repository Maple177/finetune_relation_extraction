import os
import numpy as np
import pandas as pd
import json
import logging
import torch
from transformers import BertTokenizer
from sklearn.preprocessing import MultiLabelBinarizer

logger = logging.getLogger(__name__)

class DataLoader(object):
    def __init__(self,args,tag,eval=False,inference=False):
        # tag MUST BE in {"train","dev","test"}
        self.max_seq_length = args.max_seq_length
        self.inference = inference
        self.device = args.device
        self.mlb = MultiLabelBinarizer(classes=list(range(args.num_labels)))

        data_dir = os.path.join(args.data_dir,args.task_name,"cache",args.model_name)

        wp_ids = json.load(open(os.path.join(data_dir,f"{tag}.json"),'r'))["wp_ids"]
        labels = json.load(open(os.path.join(args.data_dir,args.task_name,"label",f"{tag}.json"),'r'))["labels"]
        if not inference:
            data = list(zip(wp_ids,labels))
        else:
            data = wp_ids
        
        logger.info(f"{tag}:{len(data)} examples.")
        if args.dry_run:
            data = data[:args.number_of_examples_for_dry_run]
        
        # shuffle the data for training set if indicated        
        if args.shuffle_train:
            np.random.seed(args.seed)
            indices = list(range(len(data)))
            np.random.shuffle(indices)
            data = [data[i] for i in indices]
        
        data = [data[i:i+args.batch_size] for i in range(0,len(data),args.batch_size)]
        self.data = data
        logger.info(f"{tag}: {len(data)} batches generated.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self,key):
        if not isinstance(key,int):
            raise TypeError
        if key < 0 or key >= len(self.data):
            raise IndexError
     
        batch = self.data[key]

        if self.inference:    
            batch_wp_ids, batch_masks = self._padding(batch)        
        else:
            batch_wp_ids, batch_labels = list(zip(*batch))
            batch_wp_ids, batch_masks = self._padding(batch_wp_ids)
        
        encoding = {"input_ids":batch_wp_ids,"attention_mask":batch_masks}

        if not self.inference:
            batch_labels = self.mlb.fit_transform(batch_labels).astype(np.float32)
            encoding.update({"labels":torch.from_numpy(batch_labels).to(self.device)})
        return encoding

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    def _padding(self,wp_ids):
        max_len = max(map(len,wp_ids))
        if max_len > self.max_seq_length:
            max_len = self.max_seq_length
            wp_ids = [line[:self.max_seq_length] for line in wp_ids]	
        wp_ids = torch.Tensor([line[:max_len] + (max_len-len(line)) * [0] for line in wp_ids]).int().to(self.device)
        padded_masks = torch.Tensor([len(line) * [1] + (max_len-len(line)) * [0] for line in wp_ids]).int().to(self.device)
        return wp_ids, padded_masks
