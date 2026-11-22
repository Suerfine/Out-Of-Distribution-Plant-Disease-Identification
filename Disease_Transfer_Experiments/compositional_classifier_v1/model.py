"""Shared primitive embeddings and compositional cosine compatibility."""
import torch
from torch import nn
from torch.nn import functional as F


class Composer(nn.Module):
    def __init__(self,ncrop,ndisease,dim,hidden):
        super().__init__()
        self.crop=nn.Embedding(ncrop,dim)
        self.disease=nn.Embedding(ndisease,dim)
        nn.init.normal_(self.crop.weight,std=.02)
        nn.init.normal_(self.disease.weight,std=.02)
        self.combine=nn.Sequential(nn.Linear(2*dim,hidden),nn.ReLU(),nn.Linear(hidden,dim))

    def forward(self,crop_index,disease_index):
        c=F.normalize(self.crop(crop_index),dim=-1)
        d=F.normalize(self.disease(disease_index),dim=-1)
        return F.normalize(self.combine(torch.cat((c,d),dim=-1)),dim=-1)


class CompositionalCNN(nn.Module):
    def __init__(self,cfg,ontology,seen_compositions,pretrained=True):
        super().__init__()
        import timm
        self.net=timm.create_model(cfg['model'],pretrained=False,num_classes=0)
        if pretrained:
            from safetensors.torch import load_file
            state=load_file(cfg['pretrained_weights'])
            state.pop('head.fc.weight'); state.pop('head.fc.bias')
            self.net.load_state_dict(state,strict=True)
        dim,hidden=cfg['embedding_dim'],cfg['hidden_dim']
        self.composer=Composer(len(ontology['crop']),len(ontology['disease']),dim,hidden)
        self.image_pair=nn.Sequential(nn.Linear(self.net.num_features,hidden),nn.ReLU(),nn.Linear(hidden,dim))
        self.image_crop=nn.Linear(self.net.num_features,dim)
        self.image_disease=nn.Linear(self.net.num_features,dim)
        self.scale=cfg['cosine_scale']
        cm={v:i for i,v in enumerate(ontology['crop'])}; dm={v:i for i,v in enumerate(ontology['disease'])}
        candidates=ontology['candidates']
        self.register_buffer('pair_crop',torch.tensor([cm[p['crop_id']] for p in candidates]))
        self.register_buffer('pair_disease',torch.tensor([dm[p['disease_id']] for p in candidates]))
        self.register_buffer('seen_indices',torch.tensor([i for i,p in enumerate(candidates) if p['composition_id'] in seen_compositions]))
        raw_to_global=torch.full((max(p['composition_id'] for p in candidates)+1,),-1,dtype=torch.long)
        for i,p in enumerate(candidates): raw_to_global[p['composition_id']]=i
        self.register_buffer('raw_to_global',raw_to_global)
        local=torch.full((len(candidates),),-1,dtype=torch.long)
        local[self.seen_indices]=torch.arange(len(self.seen_indices))
        self.register_buffer('global_to_seen',local)

    def forward(self,images):
        f=self.net.forward_head(self.net.forward_features(images),pre_logits=True)
        pair_vectors=self.composer(self.pair_crop,self.pair_disease)
        return {
            'pair':self.scale*(F.normalize(self.image_pair(f),dim=1)@pair_vectors.T),
            'crop':self.scale*(F.normalize(self.image_crop(f),dim=1)@F.normalize(self.composer.crop.weight,dim=1).T),
            'disease':self.scale*(F.normalize(self.image_disease(f),dim=1)@F.normalize(self.composer.disease.weight,dim=1).T)}


def training_loss(output,labels,seen_indices,raw_to_global,global_to_seen,cfg):
    global_targets=raw_to_global[labels[:,2]]
    if (global_targets<0).any(): raise ValueError('Unknown composition in training')
    pair_targets=global_to_seen[global_targets]
    if (pair_targets<0).any(): raise ValueError('Held-out composition entered training')
    smoothing=cfg['label_smoothing']
    pair=F.cross_entropy(output['pair'][:,seen_indices],pair_targets,label_smoothing=smoothing,reduction='sum')
    crop=F.cross_entropy(output['crop'],labels[:,0],label_smoothing=smoothing,reduction='sum')
    disease=F.cross_entropy(output['disease'],labels[:,1],label_smoothing=smoothing,reduction='sum')
    total=pair+cfg['crop_loss_weight']*crop+cfg['disease_loss_weight']*disease
    return total,{'pair_ce':pair,'crop_ce':crop,'disease_ce':disease}
