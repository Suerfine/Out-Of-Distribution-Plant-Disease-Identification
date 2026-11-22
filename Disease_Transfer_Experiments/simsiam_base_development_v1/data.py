"""Label-free SSL views and deterministic augmentation for exact step resume."""
import math
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from torchvision.transforms import v2 as T, InterpolationMode
from protocol import worker_seed, transforms_for


def ssl_transform(base):
    return T.Compose([T.RandomResizedCrop(224,scale=(.7,1.),interpolation=InterpolationMode.BICUBIC,antialias=True),
        T.RandomHorizontalFlip(.5),T.RandomVerticalFlip(.5),T.ColorJitter(.2,.2,.2,0.),
        T.ToImage(),T.ToDtype(torch.float32,scale=True),T.Normalize(base['input']['mean'],base['input']['std'])])


class TrainingRows(Dataset):
    def __init__(self,rows,stage,epoch,cfg,base):
        self.rows,self.stage,self.epoch,self.cfg=rows,stage,epoch,cfg
        self.transform=ssl_transform(base) if stage=='ssl' else transforms_for(None,224)[0]

    def __len__(self): return len(self.rows)

    def __getitem__(self,index):
        row=self.rows[index]
        with Image.open(row['path']) as image: image=image.convert('RGB')
        seed=self.cfg['seed']+(0 if self.stage=='ssl' else 10_000_000)+self.epoch*100_000+index
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(seed)
            a=self.transform(image)
            if self.stage=='ssl': return a,self.transform(image),index  # No labels enter the SSL path.
        return a,int(row['disease_id']),index


def epoch_indices(count,cfg,stage,epoch):
    return np.random.default_rng(cfg['seed']+(1000 if stage=='ssl' else 2000)+epoch).permutation(count)


def training_loader(rows,stage,epoch,cursor,cfg,base):
    size=cfg['ssl_batch' if stage=='ssl' else 'finetune_batch']
    indices=epoch_indices(len(rows),cfg,stage,epoch)
    batches=[indices[i:i+size].tolist() for i in range(cursor*size,len(rows),size)]
    data=TrainingRows(rows,stage,epoch,cfg,base)
    generator=torch.Generator().manual_seed(cfg['seed']+50000+epoch)
    return DataLoader(data,batch_sampler=batches,num_workers=cfg['workers'],pin_memory=True,
        worker_init_fn=worker_seed,generator=generator,persistent_workers=False),indices
