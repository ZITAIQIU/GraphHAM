"""Base model class."""



import torch
import torch.nn as nn
import manifolds
import models.encoders as encoders
from models.decoders import model2decoder





class BaseModel(nn.Module):
    """
    Base model for graph embedding tasks.
    """

    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.manifold_name = args.manifold
        if args.c is not None:
            self.c = torch.tensor([args.c])
            if args.gpu:
                self.c = self.c.to(args.device)
        else:
            self.c = nn.Parameter(torch.Tensor([1.]))
            print('self,c cpu')
        self.manifold = getattr(manifolds, self.manifold_name)()
        if self.manifold.name == 'Hyperboloid' and args.k_i < 1:
            args.feat_dim = args.feat_dim + 1
        self.encoder = getattr(encoders, args.model)(self.c, args)

    def encode(self, x):
        if self.manifold.name == 'Hyperboloid':
            o = torch.zeros_like(x)
            x = torch.cat([o[:, 0:1], x], dim=1)
        h = self.encoder.encode(x)
        return h

    def compute_metrics(self, embeddings,  data, split):
        raise NotImplementedError

    def init_metric_dict(self):
        raise NotImplementedError

    def has_improved(self, m1, m2):
        raise NotImplementedError



class NCModel(BaseModel):
    """
    Base model for node classification task.
    """

    def __init__(self, args):
        super(NCModel, self).__init__(args)

        self.decoder = model2decoder[args.model](self.c, args)
        self.args = args

        self.f1_average = 'binary'



    def decode(self, h):
        output = self.decoder.decode(h)

        return output


    def init_metric_dict(self):
        return {'acc': -1, 'nmi': -1, 'ari': -1, 'ami': -1}

    def has_improved(self, m1, m2):
        return m1['acc'] < m2['acc']

