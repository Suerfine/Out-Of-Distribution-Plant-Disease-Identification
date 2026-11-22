"""The original Base CNN, with an optional zero-initialized local-attention branch."""
import torch
from torch import nn

from protocol import seed_all
from storage import IntegrityError
from independent_cnn_baseline.models import IndependentNet


class GatedAttention(nn.Module):
    def __init__(self,width,hidden):
        super().__init__();self.v=nn.Linear(width,hidden);self.u=nn.Linear(width,hidden);self.w=nn.Linear(hidden,1,bias=False)
    def forward(self,features):
        scores=self.w(torch.tanh(self.v(features))*torch.sigmoid(self.u(features))).squeeze(-1)
        weights=scores.float().softmax(dim=1)
        return (weights[:,:,None]*features.float()).sum(dim=1),weights


class DetailCNN(nn.Module):
    def __init__(self,base,cfg,arm,pretrained=False):
        super().__init__();net=IndependentNet(base,'disease',pretrained=pretrained)
        self.backbone,self.head,self.arm,self.local_chunk=net.backbone,net.head,arm,cfg['local_chunk']
        width=self.backbone.num_features
        if width!=1024 or sum(p.numel() for p in self.backbone.parameters())<80000000:raise IntegrityError('Expected ConvNeXt-Base')
        self.backbone.set_grad_checkpointing(cfg['grad_checkpointing'])
        if arm=='global_local':self.add_local_modules(width,cfg['attention_hidden'],175,cfg['seed']+1703)
        elif arm!='resolution384':raise IntegrityError('Unknown CNN arm')
    def add_local_modules(self,width,hidden,classes,seed):
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(seed)
            self.attention=GatedAttention(width,hidden)
            self.local_head=nn.Sequential(nn.Dropout(.2),nn.Linear(width,classes))
            nn.init.zeros_(self.local_head[1].weight);nn.init.zeros_(self.local_head[1].bias)
    def forward(self,global_view,local_views=None):
        global_logits=self.head(self.backbone(global_view))
        if self.arm=='resolution384':return global_logits
        if local_views is None or local_views.ndim!=5 or local_views.shape[1]!=4:raise IntegrityError('Four local views required')
        b,n,c,h,w=local_views.shape;flat=local_views.reshape(b*n,c,h,w)
        features=torch.cat([self.backbone(flat[i:i+self.local_chunk]) for i in range(0,len(flat),self.local_chunk)])
        pooled,_=self.attention(features.reshape(b,n,-1))
        return global_logits.float()+self.local_head(pooled).float()


def initialize(p,cfg,arm):
    seed_all(cfg['seed']);return DetailCNN(p['baseline_config'],cfg,arm,pretrained=True).to(cfg['device'])


def optimizer(model,base):
    c=base['optimizer'];groups={}
    for name,p in model.named_parameters():
        head,decay=not name.startswith('backbone.'),p.ndim>1 and not name.endswith('.bias')
        groups.setdefault((head,decay),[]).append(p)
    return torch.optim.AdamW([{'params':ps,'lr':c['lr_head' if head else 'lr_backbone'],
        'initial_lr':c['lr_head' if head else 'lr_backbone'],'weight_decay':c['weight_decay'] if decay else 0.}
        for (head,decay),ps in groups.items()],betas=tuple(c['betas']),eps=c['eps'])
