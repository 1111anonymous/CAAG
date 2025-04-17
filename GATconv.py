"""Torch modules for graph attention networks(GAT)."""
# pylint: disable= no-member, arguments-differ, invalid-name
import torch as th
import dgl
from torch import nn
from parameter_setting import parse_args
args = parse_args()
from dgl import function as fn
import torch.nn.functional as F
from dgl.nn.functional import edge_softmax
from dgl.base import DGLError
from dgl.nn.pytorch.utils import Identity
from dgl.utils import expand_as_pair

class AdaGATConv(nn.Module):
    def __init__(self,
                 in_feats,
                 edg_feats,
                 out_feats,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None,
                 allow_zero_in_degree=True,
                 bias=True):
        super(AdaGATConv, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats) # (in_feats, in_feats)
        self._edg_feats = edg_feats
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        
        self.fc = nn.Linear(
            self._in_src_feats, out_feats * num_heads, bias=False)
        self.fc_0 = nn.Linear(
            self._edg_feats, out_feats, bias=False)    
        self.fc_1 = nn.Linear(
            self._in_src_feats, out_feats, bias=False)
        self.fc_2 = nn.Linear(
            out_feats, out_feats * num_heads, bias=False)
        self.fc_src = nn.Linear(
            out_feats*2, out_feats, bias=False)
        self.fc_ada_c = nn.Linear(
            out_feats, out_feats * num_heads, bias=False)
        self.fc_ada_t = nn.Linear(
            out_feats, out_feats * num_heads, bias=False)
        self.fc_ada_d = nn.Linear(
            out_feats, out_feats * num_heads, bias=False)

        self.a_c = nn.Parameter(th.FloatTensor(1))
        self.a_t = nn.Parameter(th.FloatTensor(1))
        self.a_d = nn.Parameter(th.FloatTensor(1))

        self.attn_l = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if bias:
            self.bias = nn.Parameter(th.FloatTensor(size=(num_heads * out_feats,)))
        else:
            self.register_buffer('bias', None)
        if residual:
            if self._in_dst_feats != out_feats * num_heads:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, num_heads * out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):
        """

        Description
        -----------
        Reinitialize learnable parameters.

        Note
        ----
        The fc weights :math:`W^{(l)}` are initialized using Glorot uniform initialization.
        The attention weights are using xavier initialization method.
        """
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)
        
        nn.init.constant_(self.a_c, 0.3)
        nn.init.constant_(self.a_t, 0.3)
        nn.init.constant_(self.a_d, 0.3)

    def forward(self, graph, feat, ada_e_c, ada_e_t, ada_e_d):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph.')

            dst_prefix_shape = feat.shape[:-1] # node num
            src_prefix_shape = graph.edata['edg'].shape[:-1] # edge num
            h_src = h_dst = self.feat_drop(feat) # [node, dim]
            feat_dst = self.fc(h_src).view(*dst_prefix_shape, self._num_heads, self._out_feats) # [node, heads, out_feats]reshapes the tensor to prepare it for multi-head attention.
            feat_src = self.leaky_relu(self.fc_1(h_src)) # [node, out_feats]
            
            # We decompose the weight vector W_{gat}^T mentioned in the paper into
            # [a_l || a_r], then
            # a^T [Wh_i || Wh_j] = a_l Wh_i + a_r Wh_j + a_c mlp_out_l + a_t mlp_out_r + a_d mlp_out_d

            er = (self.leaky_relu(feat_dst) * self.attn_r).sum(dim=-1).unsqueeze(-1) # [node, heads, 1]
            graph.srcdata.update({'ft': feat_src})
            graph.dstdata.update({'er': er})
                     
            # add output of MLP to the attention score of GAT
            graph.apply_edges(lambda edges: {'e' : ((self.fc_2(edges.src['ft'] + self.fc_src(th.cat((edges.src['ft'],self.fc_0(edges.data['edg'])),1))).view(
                    *src_prefix_shape, self._num_heads, self._out_feats))* self.attn_l).sum(dim=-1).unsqueeze(-1) + edges.dst['er']}) # equ 2
            
            ada_e_c = self.fc_ada_c(ada_e_c).view(-1, self._num_heads, self._out_feats).mean(dim=-1).unsqueeze(-1)
            ada_e_t = self.fc_ada_t(ada_e_t).view(-1, self._num_heads, self._out_feats).mean(dim=-1).unsqueeze(-1)
            ada_e_d = self.fc_ada_d(ada_e_d).view(-1, self._num_heads, self._out_feats).mean(dim=-1).unsqueeze(-1)

            # hyperparameter attention a_c, a_t, a_d

            e = self.leaky_relu(graph.edata.pop('e')*th.exp(-self.a_c * ada_e_c)*th.exp(-self.a_t * ada_e_t)*th.exp(-self.a_d * ada_e_d)) # [node, heads, 1]
            
            # compute softmax
            graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))   
            graph.edata['a'] = th.where(graph.edata['a']<1e-5, th.zeros_like(graph.edata['a']), graph.edata['a'])
            
            # message passing
            graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                             fn.sum('m', 'ft')) # aggregates messages by summing them for each dst node, and stores the result in the dst node feature ft     
            rst = graph.dstdata['ft']
            # residual
            if self.res_fc is not None:
                # Use -1 rather than self._num_heads to handle broadcasting
                resval = self.res_fc(h_dst).view(*dst_prefix_shape, -1, self._out_feats)
                rst = rst + resval
            # bias
            if self.bias is not None:
                rst = rst + self.bias.view(
                    *((1,) * len(dst_prefix_shape)), self._num_heads, self._out_feats)
                
            # activation
            if self.activation:
                rst = self.activation(rst)

            return rst
