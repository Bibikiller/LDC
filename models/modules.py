from .Network import FSCILNet
from utils import *
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import cv2
import torch
from numpy.random import multivariate_normal as MultiVariateNormal
import torch.nn.functional as F

# class Sample(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, mean, cov, length):
#         return MultiVariateNormal(mean, cov, length)

#     @staticmethod
#     def backward(ctx, grad_output):
#         return grad_output.clamp_(-1, 1)
    
# def sample(mean, cov, length):
#     return Sample.apply(mean, cov, length)
    
def sample_v2(mean, cov, length):
    std_mean = np.zeros((mean.shape[0]))
    std_cov = np.ones((mean.shape[0], mean.shape[0]))
    std_samples = MultiVariateNormal(std_mean, std_cov, length)
    g_samples = mean+torch.mm(torch.tensor(std_samples).float().cuda(),cov)  # reparameterization
    return g_samples
    
def cal_mean_cov(trainset, transform, model, args):           
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

    model.module.mem_feat[:args.base_class,:] = proto_list  
    model.module.mean_cov = cov_list.mean(0).cuda()
    
    model.module.mode='ft_cos'
    
    
# generate features
def g_feat(args, model):
    if args.dataset in ['cifar100','mini_imagenet']:
        count = 60
    else:
        count=100
    model.module.g_samples = list()
    for idx in tqdm(range(count)):
        vector_exp = model.module.mem_feat[idx,:].expand(model.module.mean_cov.shape[0],model.module.mean_cov.shape[1]).unsqueeze(0).unsqueeze(0)
        cov_cat = torch.cat([vector_exp, model.module.mean_cov.unsqueeze(0).unsqueeze(0)],dim=1).float()
        mem_feat = model.module.mem_feat[idx,:]
        cov_in = model.module.layer_6(cov_cat).squeeze(0).squeeze(0)
        sam_tmp = sample_v2(mem_feat, cov_in, 30)
        model.module.g_samples.append(sam_tmp)
        
    
# distribution matching
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


