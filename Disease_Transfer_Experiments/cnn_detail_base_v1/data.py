"""Shared 384-pixel global view and four real local views from original images."""
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2 as T, functional as TF, InterpolationMode

from protocol import worker_seed
from storage import replace_with_retry


def cache_original(source,destination,size=512):
    source,destination=Path(source),Path(destination)
    with Image.open(source) as im:
        im=im.convert('RGB'); original_size=im.size; scale=size/min(im.size)
        im=im.resize((round(im.width*scale),round(im.height*scale)),Image.Resampling.BICUBIC)
        destination.parent.mkdir(parents=True,exist_ok=True); temp=destination.with_suffix('.jpg.tmp')
        im.save(temp,format='JPEG',quality=95)
    replace_with_retry(temp,destination)
    return {'original_path':str(source),'original_size':list(original_size),'cached_size':list(im.size)}


def image_views(image,base,cfg,training,include_local):
    if training:
        i,j,h,w=T.RandomResizedCrop.get_params(image,scale=tuple(base['augmentation']['random_resized_crop_scale']),ratio=(3/4,4/3))
        view=TF.crop(image,i,j,h,w)
        view=T.RandomHorizontalFlip(base['augmentation']['horizontal_flip_p'])(view)
        view=T.RandomVerticalFlip(base['augmentation']['vertical_flip_p'])(view)
        view=T.ColorJitter(**base['augmentation']['color_jitter'])(view)
    else:
        view=TF.center_crop(image,[cfg['eval_canvas'],cfg['eval_canvas']])
    def normalized(im):
        return TF.normalize(TF.to_tensor(im),mean=base['input']['mean'],std=base['input']['std'])
    global_view=normalized(TF.resize(view,[cfg['image_size'],cfg['image_size']],InterpolationMode.BICUBIC,antialias=True))
    result={'global':global_view}
    if include_local:
        canvas=TF.resize(view,[cfg['eval_canvas'],cfg['eval_canvas']],InterpolationMode.BICUBIC,antialias=True)
        size=cfg['patch_size']
        result['local']=torch.stack([normalized(TF.crop(canvas,i*size,j*size,size,size))
                                     for i in range(cfg['patch_grid']) for j in range(cfg['patch_grid'])])
    return result


class DetailImages(Dataset):
    def __init__(self,rows,base,cfg,arm,training=False,epoch=0):
        self.rows,self.base,self.cfg,self.arm,self.training,self.epoch=rows,base,cfg,arm,training,epoch
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        row=self.rows[index]
        with Image.open(row['path']) as im:image=im.convert('RGB')
        with torch.random.fork_rng(devices=[]):
            if self.training:torch.default_generator.manual_seed(self.cfg['seed']+10000000+self.epoch*100000+index)
            result=image_views(image,self.base,self.cfg,self.training,self.arm=='global_local')
        return {**result,'index':index,'label':int(row['disease_id'])}


def epoch_indices(count,cfg,epoch):
    return np.random.default_rng(cfg['seed']+2000+epoch).permutation(count)


def training_loader(p,cfg,arm,epoch,cursor):
    indices=epoch_indices(len(p['roles']['fit']),cfg,epoch);size=cfg['finetune_batch']
    batches=[indices[i:i+size].tolist() for i in range(cursor*size,len(indices),size)]
    kw={'num_workers':cfg['workers'],'pin_memory':True,'worker_init_fn':worker_seed,
        'generator':torch.Generator().manual_seed(cfg['seed']+50000+epoch),'persistent_workers':False}
    if cfg['workers']:kw['prefetch_factor']=cfg['prefetch_factor']
    return DataLoader(DetailImages(p['roles']['fit'],p['baseline_config'],cfg,arm,True,epoch),batch_sampler=batches,**kw),indices
