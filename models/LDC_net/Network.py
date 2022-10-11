# Our code is built upon the reposity https://github.com/icoz69/CEC-CVPR2021.
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet18 import *
from models.resnet20 import *
import numpy as np
from tqdm import tqdm
from utils import *

from .helper import *
from utils import *
from dataloader.data_utils import *


class LDCNET(nn.Module): 
    def __init__(self, args, mode=None): 
        super().__init__() 

        self.mode = mode 
        self.args = args 
        if self.args.dataset in ['cifar100']:  
            self.encoder = resnet20()  
            self.num_features = 64
        if self.args.dataset in ['mini_imagenet']: 
            self.encoder = resnet18(False, args)  
            self.num_features = 512
        if self.args.dataset == 'cub200':
            self.encoder = resnet18(True, args)   # pretrained=True follow TOPIC, models for cub is imagenet pre-trained. https://github.com/xyutao/fscil/issues/11#issuecomment-687548790
            self.num_features = 512
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) 
        
        if self.args.dataset in ['cifar100']:
            self.layer1 = nn.Sequential(
                nn.Linear(64, 64, bias=False),
                nn.ReLU(),
                nn.Dropout2d(p=0.5),
            )
            self.layer2 = nn.Sequential(
                nn.Linear(64, 64, bias=False),
                nn.ReLU(),
                nn.Dropout2d(p=0.5),
            )
        
            self.layer3 = nn.Linear(64, 64, bias=False)
        
            self.layer4 = nn.Sequential(
                nn.Linear(64, 64, bias=False),
                nn.ReLU(),
                nn.Dropout2d(p=0.5),
            )
            
        else:
            self.layer1 = nn.Sequential(
                nn.Linear(512, 512, bias=False),
                nn.ReLU(),
                nn.Dropout2d(p=0.5),
            )
            self.layer2 = nn.Sequential(
                nn.Linear(512, 512, bias=False),
                nn.ReLU(),
                nn.Dropout2d(p=0.5),
            )
        
            self.layer3 = nn.Linear(512, 512, bias=False)
            self.layer5 = nn.ReLU()
            self.layer4 = nn.Sequential(
                nn.Linear(512, 512, bias=False),
                nn.ReLU(),
                nn.Dropout2d(p=0.5),
            )
            
        self.layer_out2 = nn.Sequential(
            nn.Linear(512, 512, bias=False),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),
        )
            
        self.layer_out3 = nn.Sequential(
            nn.Linear(512, 512, bias=False),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),
        )
        
        self.layer_6 = nn.Sequential(
            nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1, bias=False),
        )
        
        self.fc = nn.Linear(self.num_features, self.args.num_classes, bias=False)    
        
        self.mem_mode = 'image'    
        if self.args.dataset in ['cifar100']:
            self.features = torch.zeros(100,64).cuda()
            self.covs = torch.zeros(64,64).cuda()
        else:
            self.features = torch.zeros(200,512).cuda()
            self.covs = torch.zeros(512,512).cuda()
        
        self.gate = nn.Parameter(
                torch.ones(9, device="cuda")*5,
                requires_grad=False)
        self.register_buffer('mem_feat', self.features)
        self.register_buffer('gates', self.gate)
        self.register_buffer('mean_cov', self.covs)
        self.dis_list = list()
        self.g_samples = list()
        
    # cosine or dot
    def forward_metric(self, x):
        x, g_out,_,_ = self.encode(x)
        g_label = x

        if 'cos' in self.mode:
            x = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(self.fc.weight, p=2, dim=-1))
            x = self.args.temperature * x

        elif 'dot' in self.mode:
            x = self.fc(x)

        return x, g_label

    def encode(self, x):
        x, out2, out3 = self.encoder(x)
        g_out = x
        
        ##
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.squeeze(-1).squeeze(-1)
        
        out2 = F.adaptive_avg_pool2d(out2, 1)
        out3 = F.adaptive_avg_pool2d(out3, 1)
        return x, g_out, out2, out3

    def forward(self, input):
        if self.mode != 'encoder':
            input, g_label = self.forward_metric(input)
            return input, g_label
    
        elif self.mode == 'encoder':
            input,g_out,_,_ = self.encode(input)
            return input, g_out
        else:
            raise ValueError('Unknown mode')
            
    def update_fc(self,dataloader,class_list,session,memory,data_mem,label_mem,testloader,args):
        i = 0
        for batch in dataloader:
            i+=1
            data, label = [_.cuda() for _ in batch]
            data,_,out2,out3=self.encode(data) #.detach()
            data = data.detach()
            out2 = out2.detach()
            out3 = out3.detach()
            
        origin_data = data
        print('shape of origin_data!!!!!!!!!!!', origin_data.shape)
        
        if self.args.not_data_init:
            new_fc = nn.Parameter(
                torch.rand(len(class_list), self.num_features, device="cuda"),
                requires_grad=True)
            nn.init.kaiming_uniform_(new_fc, a=math.sqrt(5))
        else:
            print('update average')
            new_fc = self.update_fc_avg(data, label, class_list)
            self.fc.weight.data[self.args.base_class + self.args.way * (session - 1):self.args.base_class + self.args.way * session, :].copy_(new_fc.data)
            
        if 'ft' in self.args.new_mode: 
            print('start finetuning')
            fc, new_fc, optimizer = self.update_fc_ft(origin_data,new_fc,data,label,session)
            

    def update_fc_avg(self,data,label,class_list):
        new_fc=[]
        fc = []
        print('check class list!!!!!!!!!', class_list)
        loop_nums = class_list[-1]+1
        print('check loop nums', loop_nums)
        for class_index in class_list:
            data_index=(label==class_index).nonzero().squeeze(-1)
            embedding=data[data_index]
            proto=embedding.mean(0)
            new_fc.append(proto)
            self.fc.weight.data[class_index]=proto
            self.mem_feat[class_index,:]=proto
            tmp_cov = torch.tensor(np.cov(embedding.cpu().numpy().T)).cuda()
            self.mean_cov = (self.mean_cov*self.args.base_class + tmp_cov*len(class_list))/(self.args.base_class+len(class_list))
        for i in class_list:
            self.g_samples.append(np.random.multivariate_normal(mean=self.mem_feat[i,:].cpu().numpy(), cov=self.mean_cov.cpu().numpy(), size=5))
        new_fc=torch.stack(new_fc,dim=0)
        return new_fc

    
    def get_logits(self,x,fc):
        if 'dot' in self.args.new_mode: 
            return F.linear(x,fc)        
        elif 'cos' in self.args.new_mode:

            norm_fc = F.normalize(fc, p=2, dim=-1)
            return self.args.temperature * F.linear(F.normalize(x, p=2, dim=-1), norm_fc)

    def update_fc_ft(self,origin_data,new_fc,data,label,session):
        new_fc=new_fc.clone().detach()
        new_fc.requires_grad=True
        fc_cls = self.fc.weight[:self.args.base_class + self.args.way * session , :].clone().detach()
        fc_cls.requires_grad=True
        optimizer = torch.optim.SGD([{'params': fc_cls, 'lr':0.01}, 
                                    ],  
                                     momentum=0.9, dampening=0.9, weight_decay=0)
        with torch.enable_grad():
            for j in range(100+10*session):
                if j<100:
                    sam_tmp = self.g_samples[j][:5,:]
                else:
                    sam_tmp = self.g_samples[j]
                g_samples = torch.Tensor(sam_tmp).cuda()
                for i in range(3):
                    g_samples = self.layer1(g_samples)
                    g_samples = self.layer2(g_samples)
                    g_samples = self.layer4(g_samples)
                g_label = j*torch.ones(g_samples.shape[0]).long().cuda()
                data = torch.cat([data,g_samples],dim=0)
                label = torch.cat([label,g_label],dim=0)
            for epoch in tqdm(range(self.args.epochs_new)):
                old_fc = self.fc.weight[:self.args.base_class + self.args.way * (session - 1), :].detach()
                fc = torch.cat([old_fc, new_fc], dim=0)
                for i in range(data.shape[0]//50):
                    logits = self.get_logits(data[50*i:50*(i+1),:],fc_cls)  
                    loss = F.cross_entropy(logits, label[50*i:50*(i+1)])  
                    optimizer.zero_grad() 
                    loss.backward(retain_graph=True) 
                    optimizer.step() 
                    pass 
        self.fc.weight.data[:self.args.base_class + self.args.way * session , :].copy_(fc_cls.data)
        return fc, new_fc, optimizer 
        
        
    def test(self, model, testloader, epoch, args, session):
        test_class = args.base_class + session * args.way
        vl = Averager()
        va = Averager()
        acc_list = list()
        with torch.no_grad():
            tqdm_gen = tqdm(testloader)
            for i, batch in enumerate(tqdm_gen, 1):
                data, test_label = [_.cuda() for _ in batch]
                logits = model(data)
                print('Logits shape in test',logits.shape)
                logits = logits[:, :test_class]
                loss = F.cross_entropy(logits, test_label)
                acc = count_acc(logits, test_label)

                vl.add(loss.item())
                va.add(acc)
                acc_list.append(acc)
            vl = vl.item()
            va = va.item()
        print('Memory {}, test, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))

        return vl, va, acc_list

        