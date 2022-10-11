from .Network import LDCNET
from utils import *
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import cv2
import torch
import torch.nn.functional as F

# replace classifier weights with prototypes
def replace_base_fc(trainset, transform, model, args):           
    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128,    
                                              num_workers=8, pin_memory=True, shuffle=False)    
    trainloader.dataset.transform = transform    
    embedding_list = []
    label_list = []
    # data_list=[]
    with torch.no_grad():
        for i, batch in enumerate(trainloader):
            data, label = [_.cuda() for _ in batch]
            model.module.mode = 'encoder'
            embedding,_ = model(data) 
            embedding_list.append(embedding.cpu()) 
            label_list.append(label.cpu()) 
    embedding_list = torch.cat(embedding_list, dim=0) 
    label_list = torch.cat(label_list, dim=0) 
    
    proto_list = []
    cov_list = []
    print('args.base_class:', args.base_class)
    for class_index in range(args.base_class):
        data_index = (label_list == class_index).nonzero()
        embedding_this = embedding_list[data_index.squeeze(-1)]
        #
        cov_this = torch.tensor(np.cov(embedding_this.cpu().numpy().T))
        embedding_this = embedding_this.mean(0)
        proto_list.append(embedding_this)
        cov_list.append(cov_this)
    proto_list = torch.stack(proto_list, dim=0) 
    cov_list = torch.stack(cov_list, dim=0) 
    print('cov_list', cov_list.shape) 

    model.module.fc.weight.data[:args.base_class] = proto_list 
    model.module.mem_feat[:args.base_class,:] = proto_list  
    model.module.mean_cov = cov_list.mean(0).cuda()
    
    model = model.train()  
    
    return model  
    
    
def g_feat(args, model):
    if args.dataset in ['cifar100','mini_imagenet']:
        count = 60
    else:
        count=100
        device='cpu'
    for idx in tqdm(range(count)):
        vector_exp = model.module.mem_feat[idx,:].expand(model.module.mean_cov.shape[0],model.module.mean_cov.shape[1]).unsqueeze(0).unsqueeze(0)
        cov_cat = torch.cat([vector_exp, model.module.mean_cov.unsqueeze(0).unsqueeze(0)],dim=1).float()
        mem_feat = model.module.mem_feat[idx,:].to(device)
        cov_in = model.module.layer_6(cov_cat).squeeze(0).squeeze(0).detach().to(device)
        sam_tmp = sample(mem_feat, cov_in, 30)
        model.module.g_samples.append(sam_tmp)
        
    

def dis_match(args,model,optimizer_rec):
    for idx,sub_list in enumerate(model.module.dis_list):
        sam_tmp = model.module.g_samples[idx]
        sam_tmp = torch.tensor(sam_tmp, dtype=torch.float32, requires_grad=True).cuda()
        sub_tensor = torch.tensor(sub_list, dtype=torch.float32, requires_grad=True).cuda()
        
        cnt=0
        loss=0
        for i in range(3):
            sam_tmp = model.module.layer1(sam_tmp)
            sam_tmp = model.module.layer2(sam_tmp)
            sam_tmp = model.module.layer4(sam_tmp)
            loss += F.kl_div(sam_tmp.softmax(dim=-1).log(), sub_tensor.softmax(dim=-1), reduction='sum')
        optimizer_rec.zero_grad()
        loss.backward(retain_graph=True)
        optimizer_rec.step()


def base_train(model, trainloader, optimizer, scheduler, epoch, args, dis_flag = False, g_label_flag = False):
    if args.dataset in ['cifar100','mini_imagenet']:
        count = 60
    else:
        count=100
    total_loss = 0
    tl = Averager()
    ta = Averager()
    model = model.train()
    # standard classification for pretrain

    tqdm_gen = tqdm(trainloader)

    for i, batch in enumerate(tqdm_gen, 1):
        data, train_label = [_.cuda() for _ in batch]
        
        logits, g_label = model(data)
        if g_label_flag and i==1:
            for i in range(count):
                model.module.dis_list.append(list())
        for i in range(train_label.shape[0]):
            if len(model.module.dis_list[train_label[i]])<30:
                model.module.dis_list[train_label[i]].append(g_label[i,:].tolist())
        logits = logits[:, :args.base_class]
        loss = F.cross_entropy(logits, train_label)
        acc = count_acc(logits, train_label)
        total_loss = loss
        
        lrc = scheduler.get_last_lr()[0]
        tqdm_gen.set_description(
            'Session 0, epo {}, lrc={:.4f},total loss={:.4f} acc={:.4f}'.format(epoch, lrc, total_loss.item(), acc))
        tl.add(total_loss.item())
        ta.add(acc)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    tl = tl.item()
    ta = ta.item()
    return tl, ta

        

def test(model, testloader, epoch, args, session):
    test_class = args.base_class + session * args.way
    model = model.eval()
    vl = Averager()
    va = Averager()
    acc_list = list()
    feat_list = list()
    with torch.no_grad():
        tqdm_gen = tqdm(testloader)
        for i, batch in enumerate(tqdm_gen, 1):
            data, test_label = [_.cuda() for _ in batch]
            logits,tsne_feat = model(data)    
            logits = logits[:, :test_class] 
            feat_list.append(logits.detach())
            loss = F.cross_entropy(logits, test_label)  
            acc = count_acc(logits, test_label)  

            vl.add(loss.item())
            va.add(acc)
            acc_list.append(acc)

        vl = vl.item()
        va = va.item()
    print('epo {}, test, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))

    return vl, va, acc_list



