import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
UNK, PAD = '<UNK>', '<PAD>'  

def build_vocab(vocab_file,is_vocab_exist=True):
    """
    构建词表
    """
    vocab_dict={}
    if is_vocab_exist:
        with open(vocab_file) as fr:
            for idx,line in enumerate(fr):
                # if idx < max_vocab_size-2:
                word=line.strip()
                vocab_dict[word]=len(vocab_dict)+1
            vocab_dict.update({UNK: len(vocab_dict), PAD: 0})
            return vocab_dict
def id2word(ids,vocab_dev):
    return "".join(vocab_dev.get(id) for id in ids if id !=0)

def category2id():
    categories = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']
    return dict(zip(categories,range(len(categories))))

class TextSet(Dataset):
    """
    构建dataset
    """
    def __init__(self,filepath,vocab,pad_size=500):
        self.sents=[]
        self.labels=[]
        label2id=category2id()
        with open(filepath) as fr:
            for line in fr:
                label,words=line.strip().split('\t')
                
                # print(label,'-----',words)
                label=label2id[label]
                words=[vocab.get(word,vocab.get(UNK)) for word in words]
                if len(words)<pad_size:
                    words.extend([0]*(pad_size-len(words)))
                else:
                    words=words[:pad_size]
                # print(label,'-----',words)
                self.sents.append(np.array(words)) # !!!切记这里不能用list
                self.labels.append(label)
            

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        sent=self.sents[index]
        label=self.labels[index]
        # sent=torch.Tensor(sent)
        # label=torch.Tensor(label)
        return sent,label


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
    vocab_file='/data02/data/corpus/cnews/cnews.vocab.txt'
    train_file='/data02/data/corpus/cnews/cnews.train.txt'
    # vocab=build_vocab(vocab_file)
    # mydataset=TextSet(train_file,vocab)
    # train_data=DataLoader(mydataset,batch_size=3,shuffle=True,num_workers=3)
    # for i,(sent,label) in enumerate(train_data):
    #     print(sent,label)
    #     break
    build_pretrain(vocab_file,'/data02/data/models/sgns.sogou.char','/data02/data/models/sgns.sogou.char.npz')
