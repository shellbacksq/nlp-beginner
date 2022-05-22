
import time
import argparse		
from aim import Run,Text

import torch
import torch.nn as nn



from tensorboardX import SummaryWriter
from sklearn import metrics
from torchinfo import summary



from dataloader import build_loader,build_vocab,TextSet,id2word,category2id
from config import TextRNNConfig,TextCNNConfig
from model import TextRNN,TextCNN



#---------------------模型参数-------------------------------#
parser = argparse.ArgumentParser()
parser.add_argument('--model',type=str,default='TextRNN',help='TextRNN or TextCNN')
args=parser.parse_args()
if args.model=='textrnn':
    config=TextRNNConfig()
    Model=TextRNN
elif args.model=='textcnn':
    config=TextCNNConfig()
    Model=TextCNN




# 准备数据
vocab_file='/data/data/corpus/cnews/cnews.vocab.txt'
train_file='/data/data/corpus/cnews/cnews.train.txt'
val_file='/data/data/corpus/cnews/cnews.val.txt'
vocab=build_vocab(vocab_file)
vocab_rev={v:k for k,v in vocab.items()}
id2category={v:k for k,v in category2id().items()}
config.vocab_size=len(vocab)

# 实验记录
# wandb.config = {
#   "learning_rate": config.lr,
#   "epochs": config.num_epochs,
#   "batch_size": config.batch_size
# }




train_dataset=TextSet(train_file,vocab,config.pad_size)
val_dataset=TextSet(val_file,vocab,config.pad_size)

train_data=build_loader(train_dataset,config)
val_data=build_loader(val_dataset,config)
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device='cpu'




def train():
    #---------------------模型实验记录-------------------------------#
    # tensorboard记录
    writer = SummaryWriter(log_dir=config.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))
    # aim记录
    run_log = Run(experiment='文本分类')
    run_log["hparams"]=config.to_dict()
    start_time=time.time()


    #-----------------------定义模型--------------------------------#
    model=Model(config)
    init_network(model)
    summary(model,(config.batch_size,config.pad_size),
                dtypes=[torch.long],
                verbose=1,
                col_width=16,
                col_names=["input_size","kernel_size", "output_size", "num_params", "mult_adds"],)

    model.to(device)
    
    optimizer=torch.optim.Adam(model.parameters(),lr=config.lr) # 定义优化器
   
    loss_func=nn.CrossEntropyLoss() # 定义损失函数
    
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
            loss=loss_func(output,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 打印
            if step%100==0:
                true = y.data.cpu()
                predic = torch.max(output.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                print('Epoch:',epoch,'|Step:',step,'|train loss:%.4f'%loss.cpu().data.numpy(),'acc',train_acc)
                
                # aim记录
                run_log.track(loss.cpu().data.numpy(), name='train_loss', epoch=epoch, context={ "subset":"train" })
                run_log.track(train_acc, name='train_acc', epoch=epoch, context={ "subset":"train" })
                run_log.track(time.time()-start_time, name='train_cost_time', epoch=epoch, context={ "subset":"train" })

                # tensorboard记录
                writer.add_scalar('loss',loss.cpu().data.numpy(),epoch)
                writer.add_scalar('acc',train_acc,epoch)
                # writer.add_graph(model,x)


                # 测试集表现
                dev_loss,dev_acc=evaluate(model,loss_func,val_data,epoch,step,run_log,writer)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(),'{}.pth'.format(args.model))
                    last_improve = total_batch  

            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break            



def evaluate(model,Loss,val_data,epoch,step,run_log,writer):
    # 评估
    model.eval()
    val_loss=0
    val_acc=0

    # 记录预测失败的文本和推理时间
    diff_x=[]
    diff_y=[]
    diff_pred=[]
    predict_cost_time=0

    for step,(x,y) in enumerate(val_data):
        x=x.to(device)
        y=y.to(device)
        output=model(x)

        if step==0:
            diff_ids=torch.argmax(output,1)!=y
            diff_x=x[diff_ids].tolist()
            diff_y=y[diff_ids].tolist()
            diff_pred=torch.argmax(output,1)[diff_ids].tolist()
            start=time.time()
            output=model(x)
            cost_time=time.time()-start
            run_log.track(cost_time/len(y)*1000, name='predict_cost_time', epoch=epoch, context={ "subset":"val" })

        loss=Loss(output,y)
        val_loss+=loss.cpu().data.numpy()
        val_acc+=metrics.accuracy_score(y.data.cpu(),torch.max(output.data,1)[1].cpu())
    val_loss/=len(val_data)
    val_acc/=len(val_data)

    # 记录信息
    print('val loss:%.4f'%val_loss,'val acc:%.4f'%val_acc)
    run_log.track(val_loss, name='val_loss', epoch=epoch, context={ "subset":"val" })
    run_log.track(val_acc, name='val_acc', epoch=epoch, context={ "subset":"val" })

    writer.add_scalar('val_loss',val_loss,epoch)
    writer.add_scalar('val_acc',val_acc,epoch)

    for x,y,z in zip(diff_x,diff_y,diff_pred):
        text=id2word(x,vocab_rev)
        text='predict:{}\t true:{}\t text:{}'.format(id2category.get(z),id2category.get(y),text)
        aim_text=Text(text)
        run_log.track(aim_text, name='bad_case', epoch=epoch, context={ "subset":"val" })

    return val_loss,val_acc

# 权重初始化，默认xavier
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