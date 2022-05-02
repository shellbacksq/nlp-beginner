class TextRNNConfig():
    def __init__(self):
        self.vocab_size=0
        self.embed_size=300
        self.hidden_size=128
        self.num_layers=2
        self.num_class=10
        self.dropout=0.5
        self.batch_size=64
        self.lr=1e-3
        self.num_epochs=10
        self.log_path='./log'

class TextCNNConfig():
    def __init__(self):
        self.vocab_size=0
        self.embed_size=300
        self.hidden_size=128
        self.num_layers=2
        self.num_class=10
        self.dropout=0.5
        self.batch_size=64
        self.lr=1e-3
        self.num_epochs=10
        self.pad_size=600
        self.log_path='./log/textcnn'