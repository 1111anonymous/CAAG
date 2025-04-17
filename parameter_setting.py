import argparse
import torch.nn.functional as F

def parse_args():
    parser = argparse.ArgumentParser(description='test recommender')
    # GAT para
    parser.add_argument('--n_heads', 
                        type=int, 
                        default=1, 
                        help='GAT heads')
    parser.add_argument('--n_out_heads', 
                        type=int, 
                        default=1, 
                        help='GAT output head')
    parser.add_argument('--n_layers', 
                        type=int, 
                        default=1, 
                        help='GAT layers')
    parser.add_argument('--n_hidden', 
                        type=int, 
                        default=120, 
                        help='hidden size')
    parser.add_argument('--activation', 
                        type=str, 
                        default= F.elu, 
                        help='activation function')
    parser.add_argument('--feat_drop', 
                        type=float, 
                        default=0.6, 
                        help='feature dropout')
    parser.add_argument('--attn_drop', 
                        type=float, 
                        default=0.6, 
                        help='attention dropout')
    parser.add_argument('--trans_drop', 
                        type=float, 
                        default=0.6, 
                        help='transformer Dropout')
    parser.add_argument('--negative_slope', 
                        type=float, 
                        default=0.2, 
                        help='negative slope')
    #attention para
    parser.add_argument('--multi_head', 
                        type=int, 
                        default=1, 
                        help='multi head in attention')
    #att_layer
    parser.add_argument('--att_layer', 
                        type=int, 
                        default=1, 
                        help='attention layer')
    #common
    parser.add_argument('--city', 
                        type=str, 
                        default='SIN', 
                        help='dataset name')
    parser.add_argument('--dim', 
                        type=int, 
                        default=120, 
                        help='dim')
    parser.add_argument('--epochs', 
                        type=int, 
                        default=100, 
                        help='epochs')
    parser.add_argument('--lr', 
                        type=float, 
                        default=1e-3, 
                        help='lr')
    parser.add_argument('--batch_size', 
                        type=int, 
                        default=32, 
                        help='batch size')
    parser.add_argument('--graph', 
                        type=str, 
                        default='transit', 
                        help='graph type: transit, cooccur, dist, tempo')
    parser.add_argument('--run_times', 
                        type=int, 
                        default=5, 
                        help='run times')
    parser.add_argument('--align_lambda', 
                        type=float, 
                        default=1, 
                        help='10, 1, 0.1, 0.01')
    parser.add_argument('--transheads', 
                        type=int, 
                        default=1, 
                        help='transformer heads')
    parser.add_argument('--trans_encoder_layers',
                        type=int,
                        default=1,
                        help='transformer encoder layers')
    
    args = parser.parse_args()

    return args
