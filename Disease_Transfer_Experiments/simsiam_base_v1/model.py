"""ConvNeXt-Base SimSiam with genuine full-batch heads and gradient caching."""
import copy
from contextlib import nullcontext
import torch
from torch import nn
from torch.nn import functional as F
from protocol import BASE_PROJECT
from independent_cnn_baseline.models import IndependentNet


class SimSiam(nn.Module):
    def __init__(self,base_config,cfg,pretrained=True):
        super().__init__()
        config=copy.deepcopy(base_config); config['model']['drop_path_rate']=0.
        self.encoder=IndependentNet(config,'disease',pretrained=pretrained).backbone
        width=self.encoder.num_features
        if width!=1024 or sum(p.numel() for p in self.encoder.parameters())<80_000_000: raise ValueError('Expected ConvNeXt-Base')
        self.projector=projector(width,cfg['projector_dim'])
        self.predictor=predictor(cfg['projector_dim'],cfg['predictor_dim'])


def projector(width,dim):
    return nn.Sequential(nn.Linear(width,width,bias=False),nn.BatchNorm1d(width),nn.ReLU(inplace=True),
        nn.Linear(width,width,bias=False),nn.BatchNorm1d(width),nn.ReLU(inplace=True),
        nn.Linear(width,dim,bias=False),nn.BatchNorm1d(dim,affine=False))


def predictor(dim,hidden):
    return nn.Sequential(nn.Linear(dim,hidden,bias=False),nn.BatchNorm1d(hidden),nn.ReLU(inplace=True),nn.Linear(hidden,dim))


def simsiam_loss(model,f1,f2):
    z1,z2=model.projector(f1),model.projector(f2)
    p1,p2=model.predictor(z1),model.predictor(z2)
    loss=-.5*(F.cosine_similarity(p1.float(),z2.detach().float(),dim=1).mean()+F.cosine_similarity(p2.float(),z1.detach().float(),dim=1).mean())
    return loss,z1,z2


def cache_backward(model,view1,view2,micro,device,amp=False,scaler=None,check_replay=False):
    """Backpropagate the exact full-batch-head objective through replayed chunks.

    The encoder must have no BatchNorm or stochastic layers. ConvNeXt uses
    LayerNorm; SSL sets DropPath/dropout to zero. All head BN layers see the
    entire 64-image view, regardless of the physical encoder microbatch.
    """
    def cast(): return torch.autocast('cuda',dtype=torch.float16) if amp else nullcontext()
    cached=[]
    for views in (view1,view2):
        chunks=[]
        with torch.no_grad():
            for start in range(0,len(views),micro):
                with cast(): f=model.encoder(views[start:start+micro].to(device,non_blocking=True))
                chunks.append(f.float())
        cached.append(torch.cat(chunks).detach().requires_grad_(True))
    with cast(): loss,z1,z2=simsiam_loss(model,*cached)
    if not torch.isfinite(loss): raise FloatingPointError('Nonfinite SSL loss')
    (scaler.scale(loss) if scaler is not None else loss).backward()
    max_error=0.
    for views,features in zip((view1,view2),cached):
        for start in range(0,len(views),micro):
            with cast(): replay=model.encoder(views[start:start+micro].to(device,non_blocking=True)).float()
            if check_replay: max_error=max(max_error,float((replay.detach()-features[start:start+micro].detach()).abs().max()))
            replay.backward(features.grad[start:start+micro])
    if check_replay and max_error!=0: raise RuntimeError(f'Gradient-cache replay differs: {max_error}')
    return float(loss.detach()),{'head_batch':len(view1),'replay_max_error':max_error}


def ssl_optimizer(model,cfg):
    return torch.optim.SGD([
        {'params':model.encoder.parameters(),'lr':cfg['ssl_backbone_lr'],'base_lr':cfg['ssl_backbone_lr'],'constant':False},
        {'params':model.projector.parameters(),'lr':cfg['ssl_projector_lr'],'base_lr':cfg['ssl_projector_lr'],'constant':False},
        {'params':model.predictor.parameters(),'lr':cfg['ssl_predictor_lr'],'base_lr':cfg['ssl_predictor_lr'],'constant':True}],momentum=.9,weight_decay=cfg['ssl_weight_decay'])
