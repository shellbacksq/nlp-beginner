import torch
import torch.nn as nn
from torchcrf import CRF

class NERLSTM_CRF(nn.Module):
    def __init__(self,config):
        super(NERLSTM_CRF,self).__init__()
        self.config=config
        if config.embedding_pretrained is not None:
            self.embedding=nn.Embedding.from_pretrained(config.embedding_pretrained,freeze=False)
        else:
            self.embedding=nn.Embedding(config.vocab_size,config.embed_size)

        self.rnn=nn.LSTM(config.embed_size,config.hidden_size,config.num_layers,dropout=config.dropout,bidirectional=True)
        self.fc=nn.Linear(config.hidden_size*2,config.target_size)
        self.crf=CRF(config.target_size)

    def forward(self,x):
        x=x.transpose(0,1)
        x=self.embedding(x)
        out,_=self.rnn(x)
        out=nn.Dropout(p=self.config.dropout)(out)
        out=self.fc(out)
        out=self.crf.decode(out)
        return out

    def log_likelihood(self, x, tags):
        x = x.transpose(0,1)
        batch_size = x.size(1)
        sent_len = x.size(0)
        tags = tags.transpose(0,1)
        x = self.embedding(x)
        out, hidden = self.rnn(x)
        out=nn.Dropout(p=self.config.dropout)(out)
        out=self.fc(out)
        return - self.crf(out, tags)