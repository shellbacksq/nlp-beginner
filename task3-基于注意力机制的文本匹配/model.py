import torch
import torch.nn as nn
import torch.nn.functional as F

class TextRNN_Att(nn.Module):
    def __init__(self,config):
        super(TextRNN_Att,self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding=nn.Embedding.from_pretrained(config.embedding_pretrained,freeze=False)
        else:
            self.embedding=nn.Embedding(config.vocab_size,config.embed_size)
        self.rnn=nn.LSTM(config.embed_size,config.hidden_size,config.num_layers,dropout=config.dropout,bidirectional=True)
        self.tanh1=nn.Tanh()
        self.w=nn.Parameter(torch.zeros(config.hidden_size*2))
        self.fc1=nn.Linear(config.hidden_size*2,config.hidden_size2)
        self.fc=nn.Linear(config.hidden_size2,config.num_class)

    def forward(self,x):
        x=self.embedding(x)
        H,_=self.rnn(x)
        H=self.tanh1(H)
        M=self.tanh1(H)

        alpha=torch.softmax(torch.matmul(M,self.w),dim=-1).unsqueeze(-1)
        out=H*alpha
        out=torch.sum(out,dim=1)
        out = F.relu(out)
        out = self.fc1(out)
        out=self.fc(out)
        return out
