
import time
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from sklearn import metrics
# from torchsummary import summary


from dataloader import build_loader,build_vocab,TextSet
from config import TextCNNConfig
from model import TextCNN

# 模型参数
config=TextCNNConfig()

# tensorboard记录
writer = SummaryWriter(log_dir=config.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))

# 准备数据
vocab_file='/data02/data/corpus/cnews/cnews.vocab.txt'
train_file='/data02/data/corpus/cnews/cnews.train.txt'
vocab=build_vocab(vocab_file)
config.vocab_size=len(vocab)
mydataset=TextSet(train_file,vocab,config.pad_size)


train_data=build_loader(mydataset,config)
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device='cpu'

def train():
    # 定义模型
    model=TextCNN(config)
    
    print(model)
    model.to(device)
    # 定义优化器
    optimizer=torch.optim.Adam(model.parameters(),lr=config.lr)
    # 定义损失函数
    loss_func=nn.CrossEntropyLoss()
    # 开始训练
    for epoch in range(config.num_epochs):
        for step,(x,y) in enumerate(train_data):
            # 训练
            x=x.to(device)
            y=y.to(device)
            output=model(x)
            # print(output,y)
            loss=loss_func(output,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 打印
            if step%100==0:
                # 每多少轮输出在训练集和验证集上的效果
                true = y.data.cpu()
                # print(true)
                # print(y.data)
                predic = torch.max(output.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                # msg = 'Epoch: {0:>6}, Step: {0:>6}, Train Loss: {},  Train Acc: {}'
                # print(msg.format(epoch, step, loss.cpu().data.numpy(), train_acc))
                print('Epoch:',epoch,'|Step:',step,'|train loss:%.4f'%loss.cpu().data.numpy(),'acc',train_acc)
                writer.add_scalar('loss',loss.cpu().data.numpy(),epoch*len(train_data)+step)
                writer.add_scalar('acc',train_acc,epoch*len(train_data)+step)
                writer.add_graph(model,x)
    # 保存模型
    torch.save(model.state_dict(),'textrnn.pth')

if __name__=='__main__':
    train()