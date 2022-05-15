
import time
import argparse		
from aim import Run

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter


from sklearn.model_selection import train_test_split
from sklearn import metrics
from torchinfo import summary

from dataloader import build_loader,build_vocab,TextSet
from config import LSTMCRFConfig
from model import NERLSTM_CRF


# 模型参数
parser = argparse.ArgumentParser()
parser.add_argument('--model',type=str,default='lstm_crf')
args=parser.parse_args()
if args.model=='lstm_crf':
    config=LSTMCRFConfig()
    Model=NERLSTM_CRF
else:
    raise ValueError('model type error')



# tensorboard记录
writer = SummaryWriter(log_dir=config.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))

# 准备数据
vocab_file='/data/data/corpus/rmrb/rmrb199801.txt'
train_file='/data/data/corpus/rmrb/train.txt'
val_file='/data/data/corpus/rmrb/valid.txt'
vocab=build_vocab(vocab_file)
config.vocab_size=len(vocab)


run_log = Run(experiment='LSTM-CRF')  
run_log["hparams"]=config.to_dict()



train_dataset=TextSet(train_file,vocab,config.pad_size)
val_dataset=TextSet(val_file,vocab,config.pad_size)

train_data=build_loader(train_dataset,config)
val_data=build_loader(val_dataset,config)

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')





def train():
    # 定义模型
    model=Model(config)
    # for name, w in model.named_parameters():
    #     print(name, '\t', w.shape)
    init_network(model)
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
    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    for epoch in range(config.num_epochs):
        for step,(x,y) in enumerate(train_data):
            # 训练
            x=x.to(device)
            y=y.to(device)
            output=model(x)
            # print(output.shape)
            loss=model.log_likelihood(x,y)
            # loss=loss_func(output,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 打印
            if step%100==0:
                # 训练集表现
                run_log.track(loss.cpu().data.numpy(), name='train_loss', step=step, context={ "subset":"train" })
                true = y.data.cpu().reshape(-1)
                output=torch.tensor(output,dtype=torch.long).reshape(-1)

                train_acc = metrics.accuracy_score(true, output)
                run_log.track(train_acc, name='train_acc', step=step, context={ "subset":"train" })
                # 验证集表现
                dev_loss,dev_acc=evaluate(model,loss_func,val_data,epoch*len(train_data)+step)
                model.train()

                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(),'{}.pth'.format(args.model))
                    last_improve = total_batch                


                print('Epoch:',epoch,'|Step:',step,'|train loss:%.4f'%loss.cpu().data.numpy(),'acc',train_acc)
                writer.add_scalar('loss',loss.cpu().data.numpy(),epoch*len(train_data)+step)
                writer.add_scalar('acc',train_acc,epoch*len(train_data)+step)

                
                # writer.add_graph(model,x)
                # 验证集表现
                # evaluate(model,loss_func,val_data,epoch)
                model.train()
            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break

    writer.close()
    # 保存模型
    torch.save(model.state_dict(),'{}.pth'.format(args.model))


def evaluate(model,Loss,val_data,idx):
    # 评估
    model.eval()
    val_loss=0
    val_acc=0
    for step,(x,y) in enumerate(val_data):
        x=x.to(device)
        y=y.to(device)
        output=model(x)
        loss=model.log_likelihood(x,y)
        # loss=Loss(output,y)
        val_loss+=loss.cpu().data.numpy()
        val_acc+=metrics.accuracy_score(y.data.cpu().reshape(-1),torch.tensor(output,dtype=torch.long).reshape(-1))
    val_loss/=len(val_data)
    val_acc/=len(val_data)
    run_log.track(val_loss, name='dev_loss', step=idx, context={ "subset":"dev" })
    run_log.track(val_acc, name='dev_acc', step=idx, context={ "subset":"dev" })
    print('val loss:%.4f'%val_loss,'val acc:%.4f'%val_acc)
    writer.add_scalar('val_loss',val_loss,idx)
    writer.add_scalar('val_acc',val_acc,idx)
    return val_loss,val_acc

#权重初始化，默认xavier
def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass


if __name__=='__main__':
    train()