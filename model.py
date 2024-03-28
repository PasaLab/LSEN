import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import CompGCNLayer


class CompGCN(nn.Module):
    def __init__(self, in_feats, n_hidden, out_feats, n_layers, activation, dropout):
        super(CompGCN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(CompGCNLayer(in_feats, n_hidden, activation, 0.))
        for i in range(n_layers - 1):
            self.layers.append(CompGCNLayer(n_hidden, n_hidden, activation, dropout))
        self.layers.append(CompGCNLayer(n_hidden, out_feats, None, dropout))
    
    def forward(self, features, relations, g):
        h = features
        for layer in self.layers:
            h = layer(h, relations, g)
        return h

class NET(nn.Module):
    def __init__(self, num_e, num_rel, num_t, args):
        super(NET, self).__init__()
        # stats
        self.num_e = num_e
        self.num_t = num_t
        self.num_rel = num_rel
        self.args = args
        self.eps = 1e-8

        # entity relation embedding
        self.rel_embeds = nn.Parameter(torch.zeros(2 * num_rel + 1, args.embedding_dim)) # rel_embeds[0] for self-loop
        nn.init.xavier_uniform_(self.rel_embeds, gain=nn.init.calculate_gain('relu'))
        self.entity_embeds = nn.Parameter(torch.zeros(self.num_e, args.embedding_dim))
        nn.init.xavier_uniform_(self.entity_embeds, gain=nn.init.calculate_gain('relu'))
        
        self.comp_gcn = CompGCN(args.embedding_dim, args.embedding_dim, args.embedding_dim, args.graph_layer, F.relu, 0.2)
        self.gru_cell = nn.GRUCell(args.embedding_dim, args.embedding_dim)

        self.linear_pred_layer_s = nn.Linear(2 * args.embedding_dim, args.embedding_dim)
        self.linear_pred_layer_o = nn.Linear(2 * args.embedding_dim, args.embedding_dim)
        
        self.dropout = nn.Dropout(args.dropout)
        self.logSoftmax = nn.LogSoftmax()
        self.softmax = nn.Softmax()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.crossEntropy = nn.BCELoss()

        print('NET Initiated')

    def forward(self, batch_block, his_g, mode_lk, total_data=None):
        quadruples, s_frequency, o_frequency = batch_block

        s = quadruples[:, 0]
        r = quadruples[:, 1]
        o = quadruples[:, 2]

        s_history_tag = copy.deepcopy(s_frequency)
        o_history_tag = copy.deepcopy(o_frequency)
       
        s_history_tag[s_history_tag != 0] = self.args.lambdax
        o_history_tag[o_history_tag != 0] = self.args.lambdax

        s_history_tag[s_history_tag == 0] = -self.args.lambdax
        o_history_tag[o_history_tag == 0] = -self.args.lambdax

        s_frequency = torch.sigmoid(s_frequency)
        o_frequency = torch.sigmoid(o_frequency)
        
        last_h, current_h = None, None
        for g in his_g:
            envolve_embs = self.comp_gcn(self.entity_embeds, self.rel_embeds, g)
            if last_h is None:
                current_h = self.gru_cell(envolve_embs)
                last_h = current_h
            else:
                current_h = self.gru_cell(envolve_embs, last_h)                
            # self.h = self.comp_gcn(self.entity_embeds, self.rel_embeds, g)
        
        if mode_lk in ['Training', 'Valid']:
            s_nce_loss, _ = self.calculate_nce_loss(s, o, r, current_h, self.rel_embeds[1:self.num_rel+1], self.linear_pred_layer_s, s_history_tag, s_frequency)
            o_nce_loss, _ = self.calculate_nce_loss(o, s, r, current_h, self.rel_embeds[self.num_rel+1:], self.linear_pred_layer_o, o_history_tag, o_frequency)
            nce_loss = (s_nce_loss + o_nce_loss) / 2.0
            
            return nce_loss

        elif mode_lk == 'Test':
            s_nce_loss, s_preds = self.calculate_nce_loss(s, o, r, current_h, self.rel_embeds[1:self.num_rel+1], self.linear_pred_layer_s, s_history_tag, s_frequency)
            o_nce_loss, o_preds = self.calculate_nce_loss(o, s, r, current_h, self.rel_embeds[self.num_rel+1:], self.linear_pred_layer_o, o_history_tag, o_frequency)
        
            sub_rank = self.link_predict(s_preds, s, o, r, total_data, 's')
            obj_rank = self.link_predict(o_preds, o, s, r, total_data, 'o')

            return sub_rank, obj_rank
        
        else:
            print("Invalid mode!")
            exit()

    def calculate_nce_loss(self, actor1, actor2, r, current_h, rel_embeds, pred_layer, history_tag, frequency):
        if current_h is not None:
            sub_emb = current_h[actor1]
            obj_emb = current_h
        else:
            sub_emb = self.entity_embeds[actor1]
            obj_emb = self.entity_embeds
        
        h = pred_layer(self.dropout(torch.cat((sub_emb, rel_embeds[r]), dim=1)))
        h = torch.tanh(h)
        preds = F.softmax(torch.mm(h, obj_emb.transpose(0, 1)) + history_tag, dim=1)
        preds = preds * frequency
        # h = pred_layer(self.dropout(torch.cat((sub_emb, rel_embeds[r]), dim=1)))
        # preds = F.softmax(torch.mm(h, obj_emb.transpose(0, 1)), dim=1)
        
        preds = preds + self.eps # avoid cross entroy loss to be nan

        nce = torch.sum(torch.gather(torch.log(preds), 1, actor2.view(-1, 1)))
        nce /= -1. * actor2.shape[0]

        return nce, preds

    def link_predict(self, preds, actor1, actor2, r, all_triples, pred_known):
        ranks = []
        for i in range(preds.shape[0]):
            cur_s = actor1[i]
            cur_r = r[i]
            cur_o = actor2[i]

            o_label = cur_o
            ground = preds[i, cur_o].clone().item()
            if self.args.filtering:
                if pred_known == 's':
                    s_id = torch.nonzero(all_triples[:, 0] == cur_s).view(-1)
                    idx = torch.nonzero(all_triples[s_id, 1] == cur_r).view(-1)
                    idx = s_id[idx]
                    idx = all_triples[idx, 2]
                else:
                    s_id = torch.nonzero(all_triples[:, 2] == cur_s).view(-1)
                    idx = torch.nonzero(all_triples[s_id, 1] == cur_r).view(-1)
                    idx = s_id[idx]
                    idx = all_triples[idx, 0]

                preds[i, idx] = 0
                preds[i, o_label] = ground

            ob_pred_comp1 = (preds[i, :] > ground).data.cpu().numpy()
            ob_pred_comp2 = (preds[i, :] == ground).data.cpu().numpy()
            ranks.append(np.sum(ob_pred_comp1) + ((np.sum(ob_pred_comp2) - 1.0) / 2) + 1)
        return ranks
