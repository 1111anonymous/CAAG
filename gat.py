import torch
import torch.nn as nn
import torch.nn.functional as F
from GATconv import AdaGATConv


class GAT( nn.Module ):
    def __init__(self,
                 g, 
                 n_layers, 
                 in_feats, 
                 edg_feats, 
                 n_hidden, 
                 heads, 
                 activation, 
                 in_drop, 
                 at_drop, 
                 negative_slope, 
                 ):
        super( GAT, self ).__init__( )
        self.g = g
        self.num_layers = n_layers
        self.activation = activation
        self.dropout = in_drop
        ####Define multi-layer GAT
        self.gat_layers = nn.ModuleList()
        self.mlp_layers_c = nn.ModuleList()
        self.mlp_layers_t = nn.ModuleList()
        self.mlp_layers_d = nn.ModuleList()
        self.gat_layers.append( AdaGATConv(
            in_feats, edg_feats, n_hidden, heads[0],
            in_drop, at_drop, negative_slope, activation=self.activation ) )
        self.mlp_layers_c.append( nn.Linear(2*in_feats, n_hidden))
        self.mlp_layers_t.append( nn.Linear(2*in_feats, n_hidden))
        self.mlp_layers_d.append( nn.Linear(2*in_feats, n_hidden))
        for l in range(1, n_layers):
            self.gat_layers.append( AdaGATConv(
                n_hidden * heads[l-1], edg_feats,  n_hidden, heads[l],
                in_drop, at_drop, negative_slope, activation=self.activation))
            self.mlp_layers_c.append( nn.Linear(n_hidden, n_hidden))
            self.mlp_layers_t.append( nn.Linear(n_hidden, n_hidden))
            self.mlp_layers_d.append( nn.Linear(n_hidden, n_hidden))
        #n_classes->in_feats
        self.gat_layers.append( AdaGATConv(
            n_hidden * heads[-2], edg_feats, in_feats, heads[-1],
            in_drop, at_drop, negative_slope, activation=None) )
        self.mlp_layers_c.append( nn.Linear(n_hidden, in_feats))
        self.mlp_layers_t.append( nn.Linear(n_hidden, in_feats))
        self.mlp_layers_d.append( nn.Linear(n_hidden, in_feats))

        

    def forward( self, inputs ):
        h = inputs
        cat_feat = self.g.ndata['cat_feat']
        time_feat = self.g.ndata['time_feat']
        src_dist_feat = self.g.ndata['src_dist_feat']
        dst_dist_feat = self.g.ndata['dst_dist_feat']
        dist_feat = self.g.edata['dist_feat']
        h_d = F.dropout(h, self.dropout, training=self.training)
        cat_feat_d = F.dropout(cat_feat, self.dropout, training=self.training)
        time_feat_d = F.dropout(time_feat, self.dropout, training=self.training)
        src_dist_feat_d = F.dropout(src_dist_feat, self.dropout, training=self.training)
        dst_dist_feat_d = F.dropout(dst_dist_feat, self.dropout, training=self.training)
        dist_feat_d = F.dropout(dist_feat, self.dropout, training=self.training)
        self.g.ndata.update({'ft_d': h_d})
        self.g.ndata.update({'cat_feat_d': cat_feat_d})
        self.g.ndata.update({'time_feat_d': time_feat_d})
        self.g.ndata.update({'src_dist_feat_d': src_dist_feat_d})
        self.g.ndata.update({'dst_dist_feat_d': dst_dist_feat_d})
        self.g.edata.update({'dist_feat_d': dist_feat_d})


        for l in range( self.num_layers ):
            
            if l == 0:
                self.g.apply_edges(lambda edges: {'ada_e_c' : self.mlp_layers_c[l](torch.cat([edges.src['cat_feat_d'],edges.dst['cat_feat_d']], dim=-1))})
                self.g.apply_edges(lambda edges: {'ada_e_t' : self.mlp_layers_t[l](torch.cat([edges.src['time_feat_d'],edges.dst['time_feat_d']], dim=-1))})
                self.g.apply_edges(lambda edges: {'ada_e_d' : self.mlp_layers_d[l](torch.cat([edges.src['src_dist_feat_d'],edges.dst['dst_dist_feat_d']], dim=-1))})
                ada_e_c = self.g.edata.pop('ada_e_c')
                ada_e_t = self.g.edata.pop('ada_e_t')
                ada_e_d = self.g.edata.pop('ada_e_d')
                ada_e_c = F.elu(ada_e_c)
                ada_e_t = F.elu(ada_e_t)
                ada_e_d = F.elu(ada_e_d)
            else:
                ada_e_c = F.dropout(ada_e_c, self.dropout, training=self.training)
                ada_e_c =  F.elu(self.mlp_layers_c[l](ada_e_c))
                ada_e_t = F.dropout(ada_e_t, self.dropout, training=self.training)
                ada_e_t =  F.elu(self.mlp_layers_t[l](ada_e_t))
                ada_e_d = F.dropout(ada_e_d, self.dropout, training=self.training)
                ada_e_d =  F.elu(self.mlp_layers_d[l](ada_e_d))
            h = self.gat_layers[l]( self.g, h, ada_e_c, ada_e_t, ada_e_d)
            h = F.elu(torch.flatten(h, start_dim=1))
            
        
        ada_e_c = F.dropout(ada_e_c, self.dropout, training=self.training)
        ada_e_c = F.elu(self.mlp_layers_c[-1](ada_e_c))
        ada_e_t = F.dropout(ada_e_t, self.dropout, training=self.training)
        ada_e_t = F.elu(self.mlp_layers_t[-1](ada_e_t))
        ada_e_d = F.dropout(ada_e_d, self.dropout, training=self.training)
        ada_e_d = F.elu(self.mlp_layers_d[-1](ada_e_d))
        logits = self.gat_layers[-1]( self.g, h, ada_e_c, ada_e_t, ada_e_d)
        logits = logits.mean( 1 )
        return logits