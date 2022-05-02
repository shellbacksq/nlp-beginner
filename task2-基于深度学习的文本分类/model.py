from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F

class TextRNN(nn.Module):
    def __init__(self,config):
        super(TextRNN,self).__init__()
        vocab_size,embed_size,hidden_size,num_layers,num_class,dropout=\
        config.vocab_size,config.embed_size,config.hidden_size,config.num_layers,config.num_class,config.dropout
        
        self.embedding=nn.Embedding(vocab_size,embed_size)
        self.rnn=nn.LSTM(embed_size,hidden_size,num_layers,bidirectional=True,batch_first=True,dropout=dropout)
        self.fc=nn.Linear(hidden_size*2,num_class)

    def forward(self,x):
        x=self.embedding(x)
        out,_=self.rnn(x)
        out=self.fc(out[:,-1,:])
        return out


class TextCNN(nn.Module):
    def __init__(self,config):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(5000, 64)
        self.conv = nn.Sequential(nn.Conv1d(in_channels=64,
                                        out_channels=256,
                                        kernel_size=5),
                              nn.ReLU(),
                              nn.MaxPool1d(kernel_size=596))

        self.f1 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.embedding(x) # batch_size x text_len x embedding_size 64*600*64
        x = x.permute(0, 2, 1) #64*64*600
        x = self.conv(x)  #Conv1后64*256*596,ReLU后不变,NaxPool1d后64*256*1

        x = x.view(-1, x.size(1)) #64*256
        x = F.dropout(x, 0.8)
        x = self.f1(x)    #64*10 batch_size * class_num
        return x
