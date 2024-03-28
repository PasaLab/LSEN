import math

import dgl.function as fn
import torch
from torch import nn


class CompGCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats, activation, dropout, bias=True):
        super(CompGCNLayer, self).__init__()
        self.weight_msg = nn.Parameter(torch.Tensor(in_feats, out_feats))
        self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_feats))
        else:
            self.bias = None
        self.activation = activation
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = 0.
        self.reset_parameters()
    
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight_msg.size(1))
        self.weight_msg.data.uniform_(-stdv, stdv)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    
    def msg_func(self, edges):
        rel_ids = edges.data['rel']
        rel_embs = self.r[rel_ids]
        node_embs = edges.src['h']
        msg = node_embs - rel_embs
        msg = torch.mm(msg, self.weight_msg)
        # rel_type = edges.data['type_rel']
        # type_weight = self.weight_msg[rel_type]
        # msg = torch.bmm(msg.unsqueeze(1), type_weight).squeeze()
        return {'msg': msg}
            
    def forward(self, h, r, g):
        self.r = r
        if self.dropout:
            h = self.dropout(h)
        # h = torch.mm(h, self.weight)
        h = h * g.ndata['norm']
        g.ndata['h'] = h
        g.update_all(lambda x: self.msg_func(x), fn.sum(msg='msg', out='h0'))
        agg_msg = g.ndata.pop('h0') * g.ndata['norm']
        h = torch.mm(g.ndata.pop('h'), self.weight)
        h = h + agg_msg
        if self.bias is not None:
            h = h + self.bias
        if self.activation:
            h = self.activation(h)
        return h
