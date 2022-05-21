import torch
import numpy as np
import re
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from config import LanguageModelConfig

BOS, EOS, UNK = '<BOS>', '<EOS>', '<UNK>'


def build_vocab(inputfile,is_vocab_exist=True):
    """
    构建词表
    """
    vocab_dict={}
    if is_vocab_exist:
        with open(inputfile) as fr:
            for idx,line in enumerate(fr):
                # if idx < max_vocab_size-2:
                word=line.strip()
                vocab_dict[word]=len(vocab_dict)+1 
    else:
        with open(inputfile) as fr:
            for idx,line in enumerate(fr):
                for word in line.strip():
                    word=word.split('/')[0]
                    if word not in vocab_dict:
                        vocab_dict[word]=len(vocab_dict)+1
    vocab_dict.update({BOS: len(vocab_dict), EOS: len(vocab_dict)+1, UNK: 0})
    return vocab_dict



class TextSet(Dataset):
    """
    构建dataset
    """
    def __init__(self,filepath,vocab,pad_size=500):
        self.sents=[]

        with open(filepath) as fr:
            sent=[]
            for idx,line in enumerate(fr):
                # if idx<=10:
                #     words=line.strip()
                #     print(words)
                if line!='\n':
                    words=line.strip()
                    words=[vocab.get(word,vocab.get(UNK)) for word in words]
                    sent.extend(words)
                elif line=='\n' and len(sent)>0:
                    if len(sent)<pad_size-2:
                        sent.extend([0]*(pad_size-2-len(sent)))
                    else:
                        sent=sent[:pad_size-2]
                    sent=[vocab.get(BOS)]+sent+[vocab.get(EOS)]

                    self.sents.append(np.array(sent)) # !!!切记这里不能用list
                    sent=[]
                # print(sent)

    def __len__(self):
        return len(self.sents)
    
    def __getitem__(self, index):
        sent=self.sents[index]
        return sent


def build_loader(dataset,config):
    dataloader=DataLoader(dataset,batch_size=config.batch_size,shuffle=True,num_workers=2)
    # dataloader=DataLoader(dataset)

    return dataloader

def build_pretrain(vocab_file,pretrain_raw,pretrain_file):
    """
    构建适配的pretrain模型
    """
    vocab_dict=build_vocab(vocab_file)
    embeddings= None

    with open(pretrain_raw,'r') as fr:
        for idx,line in enumerate(fr):
            line=line.strip().split(' ')
            if idx==0:
                embeddings=np.random.rand(len(vocab_dict),int(line[1]))
            else:
                w=line[0]
                v=line[1:]
                if w in vocab_dict:
                    emb=[float(x) for x in v]
                    embeddings[vocab_dict[w]]=np.asarray(emb,dtype='float32')
    np.savez_compressed(pretrain_file,embeddings=embeddings)



if __name__=='__main__':
    input_file='/data/data/corpus/poetry/poetryFromTang.txt'
    vocab=build_vocab(input_file,is_vocab_exist=False)
    config=LanguageModelConfig()
    train_dataset=TextSet(input_file,vocab,100)
    train_data=build_loader(train_dataset,config)

    for batch in train_data:
        print(batch)
        break