import torch
import numpy as np
class TextRNNConfig():
    def __init__(self):
        self.vocab_size=0
        self.vocab_max_size=300
        self.hidden_size=128
        self.num_layers=2
        self.num_class=10
        self.dropout=0.5
        self.batch_size=16
        self.lr=1e-3
        self.num_epochs=10
        self.pad_size=600
        self.log_path='./log'
        self.embedding_pretrained= torch.tensor(
            np.load('/data02/data/models/sgns.sogou.char.npz')["embeddings"].astype('float32')) #加载和max_vocab_size有约束
        self.embed_size = self.embedding_pretrained.size(1)\
                    if self.embedding_pretrained is not None else 300
        

class TextCNNConfig():
    def __init__(self):
        self.vocab_size=0
        self.embed_size=64
        self.hidden_size=128
        self.num_layers=2
        self.num_class=10
        self.dropout=0.5
        self.batch_size=64
        self.lr=1e-3
        self.num_epochs=10
        self.pad_size=600
        self.log_path='./log/textcnn'
        self.embedding_pretrained= torch.tensor(
            np.load('/data02/data/models/sgns.sogou.char.npz')["embeddings"].astype('float32')) #加载和max_vocab_size有约束
        self.embed_size = self.embedding_pretrained.size(1)\
                    if self.embedding_pretrained is not None else 300