import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class LanguageModel(nn.Module):
    def __init__(self,config):
        super(LanguageModel,self).__init__()
        self.config=config
        if config.embedding_pretrained is not None:
            self.embedding=nn.Embedding.from_pretrained(config.embedding_pretrained,freeze=False)
        else:
            self.embedding=nn.Embedding(config.vocab_size,config.embed_size)

        self.rnn=nn.LSTM(config.embed_size,config.hidden_size,config.num_layers,bidirectional=False)
        self.fc=nn.Linear(config.hidden_size,config.vocab_size)
        self.dropout=nn.Dropout(config.dropout)


    def forward(self,x,hidden):
        seq_len, batch_size = x.size()
        x=self.embedding(x)
        out,hidden=self.rnn(x,hidden)
        out=out.view(seq_len*batch_size,-1)
        out=self.fc(out)
        out=F.relu(out)
        return out,hidden

    def init_hidden(self, layer_num, batch_size):
        return (Variable(torch.zeros(layer_num, batch_size, self.config.hidden_size)),
                Variable(torch.zeros(layer_num, batch_size, self.config.hidden_size)))
