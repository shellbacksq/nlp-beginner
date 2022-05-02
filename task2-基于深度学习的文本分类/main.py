
import time
import argparse				#

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from sklearn import metrics
from torchinfo import summary


from dataloader import build_loader,build_vocab,TextSet
from config import TextRNNConfig,TextCNNConfig
from model import TextRNN,TextCNN

# 模型参数
parser = argparse.ArgumentParser()
parser.add_argument('--model',type=str,default='TextRNN',help='TextRNN or TextCNN')
args=parser.parse_args()
if args.model=='textrnn':
    config=TextRNNConfig()
    Model=TextRNN
elif args.model=='textcnn':
    config=TextCNNConfig()
    Model=TextCNN

# tensorboard记录
writer = SummaryWriter(log_dir=config.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))

# 准备数据
vocab_file='/data02/data/corpus/cnews/cnews.vocab.txt'
train_file='/data02/data/corpus/cnews/cnews.train.txt'
val_file='/data02/data/corpus/cnews/cnews.val.txt'
vocab=build_vocab(vocab_file)
config.vocab_size=len(vocab)

train_dataset=TextSet(train_file,vocab,config.pad_size)
val_dataset=TextSet(val_file,vocab,config.pad_size)

train_data=build_loader(train_dataset,config)
val_data=build_loader(val_dataset,config)
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device='cpu'




def train():
    # 定义模型
    model=Model(config)
    summary(model,(config.batch_size,config.pad_size),
                dtypes=[torch.long],
                verbose=1,
                col_width=16,
                col_names=["input_size","kernel_size", "output_size", "num_params", "mult_adds"],)
    # print(model)
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

                
                # writer.add_graph(model,x)
                # 测试集表现
                evaluate(model,loss_func,val_data,epoch)
                model.train()
                
    # 保存模型
    torch.save(model.state_dict(),'{}.pth'.format(args.model))


def evaluate(model,Loss,val_data,epoch):
    # 评估
    model.eval()
    val_loss=0
    val_acc=0
    for step,(x,y) in enumerate(val_data):
        x=x.to(device)
        y=y.to(device)
        output=model(x)
        loss=Loss(output,y)
        val_loss+=loss.cpu().data.numpy()
        val_acc+=metrics.accuracy_score(y.data.cpu(),torch.max(output.data,1)[1].cpu())
    val_loss/=len(val_data)
    val_acc/=len(val_data)
    print('val loss:%.4f'%val_loss,'val acc:%.4f'%val_acc)
    writer.add_scalar('val_loss',val_loss,epoch)
    writer.add_scalar('val_acc',val_acc,epoch)
    


if __name__=='__main__':
    train()