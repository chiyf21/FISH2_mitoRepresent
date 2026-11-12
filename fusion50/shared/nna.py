"""Use the existing official MIM-Refiner NNA loss, queue and MLP builders unchanged."""
import ast,importlib.util,sys,hashlib
from pathlib import Path
import torch
from torch import nn

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def components(root):
    root=Path(root);sys.path.append(str(root))
    import utils
    if str(root/'utils') not in utils.__path__:utils.__path__.append(str(root/'utils'))
    loss=module('_pipeline_nna_loss',root/'losses/ssl/nnclr_loss.py').NnclrLoss
    queue=module('_pipeline_nna_queue',root/'modules/ssl/nnclr_queue.py').NnclrQueue
    tree=ast.parse((root/'models/ssl/nnclr_head.py').read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='NnclrHead')
    methods=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ('create_projector','create_predictor')]
    scope={'nn':nn};exec(compile(ast.Module(body=methods,type_ignores=[]),str(root/'models/ssl/nnclr_head.py'),'exec'),scope)
    return loss,queue,scope

def hashes(root):
    return {p:hashlib.sha256((Path(root)/p).read_bytes()).hexdigest() for p in ('losses/ssl/nnclr_loss.py','modules/ssl/nnclr_queue.py','models/ssl/nnclr_head.py')}

class Head(nn.Module):
    def __init__(self,queue,builders,cfg):
        super().__init__()
        from kappamodules.utils.mode_to_ctor import mode_to_norm_ctor
        from kappamodules.init import init_norm_as_noaffine,init_xavier_uniform_zero_bias
        self.input_dim=384;self.proj_hidden_dim=2048;self.proj_hidden_layers=1
        self.output_dim=256;self.pred_hidden_dim=4096;self.pred_hidden_layers=0
        self.norm_ctor,self.requires_bias=mode_to_norm_ctor('batchnorm');self.act_ctor=nn.GELU
        self.projector=builders['create_projector'](self);self.predictor=builders['create_predictor'](self)
        self.queue=queue(size=cfg['queue_size'],dim=256,topk=cfg['topk'],guidance='none')
        self.apply(init_norm_as_noaffine);self.apply(init_xavier_uniform_zero_bias)
    def forward(self,features,ids):
        b,v,d=features.shape
        if v!=2:raise ValueError('Point pipeline uses exactly two augmented views')
        projected=self.projector(features.transpose(0,1).reshape(v*b,d));predicted=self.predictor(projected)
        target=projected.detach();neighbor,_=self.queue(x=target,idx=ids,cls=None)
        return dict(projected_preswap=target.reshape(v,b,256).transpose(0,1),projected_postswap=neighbor.reshape(v,b,256).transpose(0,1),predicted=predicted.reshape(v,b,256).transpose(0,1))

class Supervision(nn.Module):
    def __init__(self,root,cfg):
        super().__init__();loss,queue,builders=components(root)
        self.heads=nn.ModuleDict({str(l):Head(queue,builders,cfg) for l in cfg['layers']});self.loss=loss(temperature=cfg['temperature'])
    def forward(self,features,ids):
        parts={k:self.loss(**head(features[int(k)],ids))[0]['total'] for k,head in self.heads.items()}
        return sum(parts.values())/len(parts),parts
