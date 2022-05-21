import torch
import numpy as np
import re
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
UNK, PAD = '<UNK>', '<PAD>'  

def BMEWO(sent):
    w_t_dict = {'t':'_TIME','nr':'_PERSON','ns':'_LOCATION','nt':'_ORGANIZATION'}
    charLst = []
    tagLSt = []

    sent_split_lst = sent.strip().split("  ")
    for w in sent_split_lst:
        if "/" in w:
            ww = w.split("/")[0]
            wtag = w.split("/")[1]
            ww2char = [c for c in ww]
            charLst.extend(ww2char)
            #获得词的长度
            wlen = len(ww)
            #初始化标注
            #其他标注成'O'
            BMEWO_tag = ['O' for _ in ww]

            #如果是['t','nr','ns','nt'] 标注
            if wtag in ['t','nr','ns','nt']:
                BMEWO_tag[0] = 'B' + w_t_dict[wtag]
                BMEWO_tag[-1] = 'E' + w_t_dict[wtag]
                BMEWO_tag[1:-1] = ['M' + w_t_dict[wtag] for _ in BMEWO_tag[1:-1]]
            tagLSt.extend(BMEWO_tag)
        else:
            continue
    return charLst,tagLSt

def strQ2B(ustring):
    """全角转半角"""
    rstring = ""
    for uchar in ustring:
        inside_code=ord(uchar)
        if inside_code == 12288:                              #全角空格直接转换
            inside_code = 32
        elif (inside_code >= 65281 and inside_code <= 65374): #全角字符（除空格）根据关系转化
            inside_code -= 65248
        rstring += chr(inside_code)
    return rstring

def deal_(text_line):
    # 合并中括号里面的内容
    blacketLst = re.findall("\[.*?\]", text_line)
    if blacketLst:
        for b in blacketLst:
            b_trans = "".join([i.split("/")[0] for i in b[1:-1].split("  ")]) + "/"
            text_line = text_line.replace(b, b_trans)
    # 将姓名空格去除
    text_line = re.sub("(\w+)/nr  (\w+)/nr", r"\1\2/nr", text_line)
    # 将时间空格去除
    text_line = re.sub("(\w+)/t  (\w+)/t", r"\1\2/t", text_line)
    text_line = re.sub("(\w+)/t  (\w+)/t  (\w+)/t", r"\1\2\3/t", text_line)
    text_line = re.sub("(\w+)/t  (\w+)/t  (\w+)/t  (\w+)/t", r"\1\2\3\4/t", text_line)

    #去除（/w ）/w
    text_line = re.sub("（/w  ","",text_line)
    text_line = re.sub("  ）/w", "", text_line)
    return text_line

def trans_rmrb(inpath,out_dir):
    with open(inpath) as fr:
        char_data=[]
        tag_data=[]
        for line in fr:
            if line:
                line=line.strip()[23:]
                line_list = line.split("。/w  ") #分句
                
                for sent in line_list:
                    sent=sent.strip()
                    if len(sent)>20:
                        sent=deal_(sent)
                        sent=strQ2B(sent)
                        charLst,tagLSt = BMEWO(sent)
                        char_data.append(charLst)
                        tag_data.append(tagLSt)
        print(set([tag_data[i][j] for i in range(len(tag_data)) for j in range(len(tag_data[i]))]))
        x_train,x_test, y_train, y_test = train_test_split(char_data, tag_data, test_size=0.2, random_state=43)
        x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train,  test_size=0.2, random_state=43)
        with open(out_dir+"/train.txt",'w') as fw:
            for i in range(len(x_train)):
                fw.write(" ".join(x_train[i])+"\t"+" ".join(y_train[i])+"\n")
        with open(out_dir+"/valid.txt",'w') as fw:
            for i in range(len(x_valid)):
                fw.write(" ".join(x_valid[i])+"\t"+" ".join(y_valid[i])+"\n")  
        with open(out_dir+"/test.txt",'w') as fw:
            for i in range(len(x_valid)):
                fw.write(" ".join(x_test[i])+"\t"+" ".join(y_test[i])+"\n")   




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
                for word in line.strip().split()[1:]:
                    word=word.split('/')[0]
                    if word not in vocab_dict:
                        vocab_dict[word]=len(vocab_dict)+1
    vocab_dict.update({UNK: len(vocab_dict), PAD: 0})
    return vocab_dict

def category2id():
    categories = ['E_ORGANIZATION', 'B_LOCATION', 'E_LOCATION', 'E_TIME', 'O', 'M_PERSON', 'B_TIME', 'E_PERSON', 'M_LOCATION', 'B_PERSON', 'M_TIME', 'B_ORGANIZATION', 'M_ORGANIZATION']
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
                words,tag=line.strip().split('\t')
                
                # print(label,'-----',words)
                tags=[label2id[tag] for tag in tag.split()]
                words=[vocab.get(word,vocab.get(UNK)) for word in words.split()]
                if len(words)<pad_size:
                    words.extend([0]*(pad_size-len(words)))
                    tags.extend([label2id['O']]*(pad_size-len(tags)))
                else:
                    words=words[:pad_size]
                    tags=tags[:pad_size]
                # print(label,'-----',words)
                self.sents.append(np.array(words)) # !!!切记这里不能用list
                self.labels.append(np.array(tags))
            

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

    input_file='/data/data/corpus/rmrb/rmrb199801.txt'
    output_file='/data/data/corpus/rmrb/rmrb199801.txt.process'
    out_dir='/data/data/corpus/rmrb/'
    # vocab=build_vocab(vocab_file)
    # mydataset=TextSet(train_file,vocab)
    # train_data=DataLoader(mydataset,batch_size=3,shuffle=True,num_workers=3)
    # for i,(sent,label) in enumerate(train_data):
    #     print(sent,label)
    #     break
    # build_pretrain(vocab_file,'/data02/data/models/sgns.sogou.char','/data02/data/models/sgns.sogou.char.npz')
    # d=build_vocab(input_file,is_vocab_exist=False)
    # print(d)
    trans_rmrb(input_file,out_dir)