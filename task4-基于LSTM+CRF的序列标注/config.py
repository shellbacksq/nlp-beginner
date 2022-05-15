import torch
import numpy as np
from dataloader import category2id
class LSTMCRFConfig(object):
    def __init__(self):
        self.vocab_size=0
        self.vocab_max_size=300
        self.hidden_size=128
        self.num_layers=2
        self.num_class=10
        self.dropout=0.5
        self.batch_size=64
        self.lr=1e-3
        self.num_epochs=5
        self.pad_size=60
        self.log_path='./log'
        self.embedding_pretrained=None
        # self.embedding_pretrained= torch.tensor(
        #     np.load('/data02/data/models/sgns.sogou.char.npz')["embeddings"].astype('float32')) #加载和max_vocab_size有约束
        self.embed_size = self.embedding_pretrained.size(1)\
                    if self.embedding_pretrained is not None else 300
        self.hidden_size2=64
        self.require_improvement = 1000   
        self.target_size=len(category2id())

    def to_dict(self):
        res=self.__dict__.copy()
        res.pop('embedding_pretrained')
        return res