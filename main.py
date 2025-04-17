import pickle
import dgl
import torch
from torch import nn
import logging, os
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from parameter_setting import parse_args
from result_process import cal_ave_result
import torch.utils.data as data
from data import Mydata
from gat import GAT

if torch.cuda.is_available():
    device = 'cuda'
# elif torch.backends.mps.is_available():
#     device = 'mps'
else:
    device = 'cpu'
device = 'cpu'
save_model = True


class ModelSaver:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        
    def save_checkpoint(self, model, epoch, optimizer, loss, metrics, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'metrics': metrics
        }
        
        # Save regular checkpoint
        # checkpoint_path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch}.pth')
        # torch.save(checkpoint, checkpoint_path)
        
        # Save best model if this is the best performance
        if is_best:
            best_path = os.path.join(self.save_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            logging.info(f'Saved best model to {best_path}')

    def load_checkpoint(self, model, optimizer=None, checkpoint_path=None):
        """Load model from checkpoint"""
        if checkpoint_path is None:
            # Load best model by default
            checkpoint_path = os.path.join(self.save_dir, 'best_model.pth')
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
            
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        return checkpoint['epoch'], checkpoint['loss'], checkpoint['metrics']



class model( nn.Module ):
    def __init__(self,
                 g, #DGL graph
                 embeddings, # embedding dict
                 n_layers, # GAT layer
                 in_feats, # POI dim
                 edg_feats, # edg dim
                 n_hidden, # hidden dim
                #  n_classes,
                 heads, # list: head number in each GAT layer
                 activation,
                 in_drop, # feature dropout ratio
                 at_drop, # attention dropout ratio
                 negative_slope, # a para of Leakyrelu
                 max_seq, # max seq length
                 multi_head,
                 epochs,
                 lr,
                 att_layer,
                 align_lambda,
                 trans_drop,
                 transheads,
                 trans_encoder_layers
                 ):
        super( model, self ).__init__( )
        self.embedding = embeddings
        self.dropout = in_drop
        self.g = g
        self.epochs = epochs
        self.lr = lr
        self.lamda = align_lambda
        self.att_layer = att_layer
        self.gat = GAT( g,
                 n_layers,
                 in_feats,
                 edg_feats,
                 n_hidden,
                 heads,
                 activation,
                 in_drop,
                 at_drop,
                 negative_slope)
        self.total_embed_size = 5*in_feats
        self.encoder = nn.Transformer(
                                        d_model=self.total_embed_size,
                                        nhead=transheads,
                                        num_encoder_layers=trans_encoder_layers,
                                        num_decoder_layers=1,
                                        dim_feedforward=self.total_embed_size * 2, #forward_expansion=2
                                        dropout=trans_drop,
                                        batch_first=True
                                    )
        self.layers = nn.ModuleList(
            [nn.MultiheadAttention(
            embed_dim=in_feats,
            num_heads=multi_head,)
            for _ in range(1, self.att_layer) 
            ]
        )
        self.att1 = nn.MultiheadAttention(
            embed_dim=in_feats,
            num_heads=multi_head,
        )
        self.embeds_n = self.embedding['node']
        self.user_embed = self.embedding['user']
        self.fc_g = nn.Linear(in_feats, len(g.nodes()))
        self.fc_h = nn.Linear(self.total_embed_size, in_feats)
        self.fc_s = nn.Linear(self.total_embed_size, len(g.nodes()))
        

    def forward(self, user, seq, cat, hour, weekday):
        feature = self.g.ndata['feat']
        h_p = self.gat(feature) # h_p: [len(g.nodes()), dim]
        # the graph updated POI representation
        h_g_all = h_p[seq[:,:]] # all the sequence [bs, max_seq, dim]

        # Transformer Encoder
        poi_emb = torch.stack([feature[seq[x]] for x in range(seq.shape[0])])# all the sequence  [bs, max_seq, dim]
        user_emb = self.user_embed(user) # [bs, 1, dim]
        # expand user_emb as the same dimention of  poi_emb dim=1 
        user_emb_expanded = user_emb.expand(-1, poi_emb.size(1), -1)  # (bs, max_seq, dim)
        cat_emb = self.embedding['category'](cat) # [bs, max_seq, dim]
        hour_emb = self.embedding['time'](hour) # [bs, max_seq, dim]
        weekday_emb = self.embedding['weekday'](weekday) # [bs, max_seq, dim]
        # concat at dim=2 
        combined_emb = torch.cat((poi_emb, user_emb_expanded, cat_emb, hour_emb, weekday_emb), dim=2)  # (bs, max_seq, 5 * dim)
        src = combined_emb
        # Create a mask
        src_mask = self.encoder.generate_square_subsequent_mask(src.shape[1]).to(src.device)
        output = self.encoder(src, src, src_mask=src_mask, tgt_mask=src_mask) # (bs, max_seq, 2 * dim)
        h_s = output[:,-1,:] # [bs, 2*dim]
        h_poi_s = self.fc_s(F.dropout(h_s, self.dropout, training=self.training))  # sequential rep [bs,len(g.nodes())]
        logits = F.elu(h_poi_s) # [bs, len(g.nodes())]
        
        # all node KL alignment
        h_s_all = output[:,:,:] # [bs, max_seq, 2dim]
        h_s_all = self.fc_h(F.dropout(h_s_all, self.dropout, training=self.training)) # [bs, max_seq, dim]
        
        return logits, h_s_all, h_g_all
    
    def fit(self, train_loader, test_loader, file_name):
        model_saver = ModelSaver(file_name)
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-5)
        patience_limit = 5  
        best_val_epoch, best_val_ndcg, best_val_recall = 0,  0., 0.
        best_recall_1, best_NDCG_1, best_recall_5, best_NDCG_5, best_recall_10, best_NDCG_10 = 0., 0., 0., 0., 0., 0.
        result_save_path = file_name + f'.txt'
        for epoch in range(1, self.epochs + 1):
            self.train()
            current_loss = 0.
            align_loss = 0.
            sample_num = 0
            # set process bar display
            pbar = tqdm(train_loader)
            pbar.set_description(f'[Epoch {epoch:03d}]')
            for user, seq, label, cat, hour, weekday in pbar:
                self.zero_grad()
                logits, h_poi_s, h_poi_g = self.forward(user, seq, cat, hour, weekday) # [bs, len(g.nodes())]
                # pred loss
                label = label.squeeze(1) # [bs, 1]    
                loss_fcn = torch.nn.CrossEntropyLoss()
                loss_pred = loss_fcn( logits, label )
                # reg loss
                l2_reg = torch.tensor(0., requires_grad=True)
                for param in self.parameters():
                    l2_reg = l2_reg + torch.norm(param)
                # KL mutual loss
                loss_sg = F.kl_div(F.log_softmax(h_poi_s, dim=-1), F.softmax(h_poi_g, dim=-1), reduction='mean')
                loss_gs = F.kl_div(F.log_softmax(h_poi_g, dim=-1), F.softmax(h_poi_s, dim=-1), reduction='mean')
                # pred loss + align loss + L2 regularization
                loss = loss_pred + self.lamda * (loss_sg + loss_gs) + 1e-5 * l2_reg
        
                if torch.isnan(loss).any():
                    if torch.isnan(logits).any():
                        print("Logits contain NaN values.")
                    if torch.isinf(logits).any():
                        print("Logits contain Inf values.")
                loss.backward(retain_graph=True)
                optimizer.step() 
                current_loss += loss.item()
                sample_num+=len(seq)
            logging.info(f'Epoch: {epoch:04d}, sample_num: {sample_num:04d}, loss_train: {current_loss/sample_num:.4f}')

            # Evaluate the model
            self.eval()           
            recall_1, NDCG_1, MAP_1, recall_5, NDCG_5, MAP_5, recall_10, NDCG_10, MAP_10 = self.test(test_loader, epoch, best_val_epoch, file_name)
            
            metrics = {
                        'recall_1': recall_1, 'recall_5': recall_5, 'recall_10': recall_10,
                        'ndcg_1': NDCG_1, 'ndcg_5': NDCG_5, 'ndcg_10': NDCG_10,
                        'map_1': MAP_1, 'map_5': MAP_5, 'map_10': MAP_10
                        }
            
            is_best = False
            if NDCG_10 > best_val_ndcg or epoch == 1:
                best_val_epoch, best_val_ndcg, best_val_recall = epoch, NDCG_10, recall_10
                best_recall_1, best_NDCG_1, best_recall_5, best_NDCG_5, best_recall_10, best_NDCG_10 = recall_1, NDCG_1, recall_5, NDCG_5, recall_10, NDCG_10
                is_best = True
                
                with open(result_save_path,'w') as f:        
                    for k in [1,5,10]:
                        if k ==1:
                            f.write(f"HR @{k} : {recall_1},\tNDCG@{k} : {NDCG_1},\tMAP@{k} : {MAP_1} \n")
                            logging.info(f"HR @{k} : {recall_1},\tNDCG@{k} : {NDCG_1},\tMAP@{k} : {MAP_1}")
                        if k ==5:
                            f.write(f"HR @{k} : {recall_5},\tNDCG@{k} : {NDCG_5},\tMAP@{k} : {MAP_5} \n")
                            logging.info(f"HR @{k} : {recall_5},\tNDCG@{k} : {NDCG_5},\tMAP@{k} : {MAP_5}")
                        if k ==10:
                            f.write(f"HR @{k} : {recall_10},\tNDCG@{k} : {NDCG_10},\tMAP@{k} : {MAP_10} \n")
                            logging.info(f"HR @{k} : {recall_10},\tNDCG@{k} : {NDCG_10},\tMAP@{k} : {MAP_10}")
                    f.close()

            model_saver.save_checkpoint(
                                        model=self,
                                        epoch=epoch,
                                        optimizer=optimizer,
                                        loss=current_loss/sample_num,
                                        metrics=metrics,
                                        is_best=is_best
                                        )
            if epoch - best_val_epoch == patience_limit:
                logging.info(f'Stop training after {patience_limit} epochs without valid improvement.')
                logging.info(f'Test HR@1: {best_recall_1:.5f}, HR@5: {best_recall_5:.5f}, HR@10: {best_recall_10:.5f}, NDCG@1: {best_NDCG_1:.5f}, NDCG@5: {best_NDCG_5:.5f}, NDCG@10: {best_NDCG_10:.5f}')
                break
            logging.info(f'Best valid NDCG@10 at epoch {best_val_epoch}')
            

    def predict(self,user,seq,cat, hour, weekday):
        logits, _, _ = self.forward(user, seq, cat, hour, weekday)
        ranking = torch.sort(logits, descending=True)[1]
        return ranking

    def test(self, test_loader, epoch, best_val_epoch, file_name, ks=[1,5,10]):

        def calc_recall(labels, preds, k):
            return torch.sum(torch.sum(labels==preds[:,:k], dim=1))/labels.shape[0]
        
        def calc_ndcg(labels, preds, k):
            exist_pos = (preds[:,:k] == labels).nonzero()[:,1] + 1
            ndcg = 1/torch.log2(exist_pos+1)
            return torch.sum(ndcg) / labels.shape[0]

        def calc_map(labels, preds, k):
            exist_pos = (preds[:,:k] == labels).nonzero()[:,1] + 1
            map = 1/exist_pos
            return torch.sum(map) / labels.shape[0]

        recalls_1, NDCGs_1, MAPs_1 = [],[],[]
        recalls_5, NDCGs_5, MAPs_5 = [],[],[]
        recalls_10, NDCGs_10, MAPs_10 = [],[],[]
        recall_1, NDCG_1, MAP_1 = 0.0, 0.0, 0.0
        recall_5, NDCG_5, MAP_5 = 0.0, 0.0, 0.0
        recall_10, NDCG_10, MAP_10 = 0.0, 0.0, 0.0
        pbar = tqdm(test_loader)
        for user, seq, label, cat, hour, weekday in pbar:
            pred = self.predict(user,seq,cat, hour, weekday)
            for k in ks:
                if k==1:
                    recalls_1.append(calc_recall(label, pred, k))
                    NDCGs_1.append(calc_ndcg(label, pred, k))
                    MAPs_1.append(calc_map(label, pred, k))
                if k==5:
                    recalls_5.append(calc_recall(label, pred, k))
                    NDCGs_5.append(calc_ndcg(label, pred, k))
                    MAPs_5.append(calc_map(label, pred, k))
                if k==10:
                    recalls_10.append(calc_recall(label, pred, k))
                    NDCGs_10.append(calc_ndcg(label, pred, k))
                    MAPs_10.append(calc_map(label, pred, k))
        recall_1 = torch.stack(recalls_1).mean()
        NDCG_1 = torch.stack(NDCGs_1).mean()
        MAP_1 = torch.stack(MAPs_1).mean()
        recall_5 = torch.stack(recalls_5).mean()
        NDCG_5 = torch.stack(NDCGs_5).mean()
        MAP_5 = torch.stack(MAPs_5).mean()
        recall_10 = torch.stack(recalls_10).mean()
        NDCG_10 = torch.stack(NDCGs_10).mean()
        MAP_10 = torch.stack(MAPs_10).mean()
            
        return recall_1, NDCG_1, MAP_1, recall_5, NDCG_5, MAP_5, recall_10, NDCG_10, MAP_10


def set_seed(seed: int = 888):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_logging(log_file_path):
    logger = logging.getLogger()

    if not logger.hasHandlers():  # Check if the logger has been configured
        logger.setLevel(logging.INFO)

        file_handler = logging.FileHandler(log_file_path, mode='w')
        file_handler.setLevel(logging.INFO)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

if __name__ == '__main__':
    set_seed(3407)

    args = parse_args()

    if args.city == 'NY':
        # NY param
        args.lr = 1e-4
        args.dim = 150
        args.align_lambda = 1 
        args.trans_drop = 0.7 
        args.n_head = 3 
        args.n_layers = 5 
        args.transheads = 1
        args.trans_encoder_layers = 2 

    elif args.city == 'PHO':
        # PHO param
        args.lr = 1e-4
        args.dim = 180
        args.align_lambda = 0.7 
        args.trans_drop = 0.4 
        args.n_head = 3 
        args.n_layers = 3 
        args.transheads = 1
        args.trans_encoder_layers = 5 

    elif args.city == 'SIN':
        # SIN param
        args.lr = 5e-5
        args.dim = 150
        args.align_lambda = 1 
        args.trans_drop = 0.7 
        args.n_head = 3 
        args.n_layers = 2 
        args.transheads = 3
        args.trans_encoder_layers = 2 
    
    file_path = f'./result/{args.city}_{args.epochs}'
    for num in range(args.run_times):
        file_name = file_path + f'_{num}'
        os.makedirs('./logs/', exist_ok=True)
        log_file_path = os.path.join('./logs/', f'training_{args.city}_{num}.log')
        setup_logging(log_file_path)

        # %% ====================== Load data ======================
        logging.info(f'city: {args.city}, lr: {args.lr}, n_hidden: {args.n_hidden}, dim: {args.dim}, batch_size: {args.batch_size}, n_layers: {args.n_layers}, n_heads: {args.n_heads}, transheads: {args.transheads}, trans_encoder_layers: {args.trans_encoder_layers}, epoch:{args.epochs}, KL_mutual_weights:{args.align_lambda}')
        with open(f'./data/{args.city}_metadata_{args.graph}.pkl', 'rb') as file:
                meta_data = pickle.load(file)
        user_num = len(meta_data['UserId'].keys())
        max_seq = meta_data['max_seq']
        train_data = Mydata(args.city, 0, max_seq, 'train', device)
        test_data = Mydata(args.city, 0, max_seq, 'test', device)
        
        (g,), _ = dgl.load_graphs((f'./data/{args.city}_train_graph_{args.graph}.dgl'))
        g = g.to(device)
        node = g.nodes()
        '''
        g.edges() returns a tuple (src, dst);  
        g.edges(form = 'all') returns a tuple (src, dst, eid), 
        g.edges(form = 'all')[2] specifically selecting the eid tensor.
        '''
        edge = g.edges(form = 'all')[2] 

        # %% ====================== initial embedding ======================
        embeddings = {
                        'user': torch.nn.Embedding(user_num + 1, args.dim).to(device),
                        'node': torch.nn.Embedding(len(node), args.dim, padding_idx=0).to(device),
                        'category': torch.nn.Embedding(max(g.ndata['cat']).item() + 1, args.dim, padding_idx=0).to(device),
                        'edge': torch.nn.Embedding(len(edge), args.dim).to(device),
                        'frequency': torch.nn.Embedding(max(g.edata['freq']).item() + 1, args.dim).to(device),
                        'distance': torch.nn.Embedding(meta_data['max_dist'] + 1, args.dim).to(device),
                        'time': torch.nn.Embedding(24, args.dim, padding_idx=0).to(device),
                        'weekday': torch.nn.Embedding(2, args.dim, padding_idx=0).to(device)
                    }

        g.ndata['feat'] = embeddings['node'](node) # all nodes' embedding [len(node), dim]
        g.ndata['cat_feat'] = embeddings['category'](g.ndata['cat'])
        src_dist_freq = g.ndata['src_dist_freq'].to(embeddings['distance'].weight.dtype)
        dst_dist_freq = g.ndata['dst_dist_freq'].to(embeddings['distance'].weight.dtype)
        g.ndata['src_dist_feat'] = torch.matmul(src_dist_freq, embeddings['distance'].weight) # [len(node), dim]
        g.ndata['dst_dist_feat'] = torch.matmul(dst_dist_freq, embeddings['distance'].weight) # [len(node), dim]
        g.ndata['time_feat'] = torch.matmul(g.ndata['time_freq'].to(embeddings['time'].weight.dtype), embeddings['time'].weight) # [len(node), dim]
        g.edata['edg'] = embeddings['edge'](edge)
        g.edata['freq_feat'] = embeddings['frequency'](g.edata['freq'])
        g.edata['dist_feat'] = embeddings['distance'](g.edata['dist'])
        
        # %% ====================== initial model ======================
        in_feats = args.dim
        edg_feats = args.dim
        n_classes = g.num_nodes()
        heads = ([args.n_heads] * args.n_layers) + [args.n_out_heads] # head number of each layer
        Model = model(  g,
                        embeddings,
                        args.n_layers,
                        in_feats,
                        edg_feats,
                        args.n_hidden,
                        heads,
                        args.activation,
                        args.feat_drop,
                        args.attn_drop,
                        args.negative_slope,
                        max_seq,
                        args.multi_head,
                        args.epochs,
                        args.lr,
                        args.att_layer,
                        args.align_lambda,
                        args.trans_drop,
                        args.transheads,
                        args.trans_encoder_layers
                        )
        Model.to(device)
        train_loader = data.DataLoader(
                train_data, 
                batch_size=args.batch_size,
                shuffle=True
            )
        test_loader = data.DataLoader(
                test_data, 
                batch_size=args.batch_size,
                shuffle=False
            )

        # %% ====================== train ======================
        Model.fit(train_loader, test_loader, file_name)
        # # %% ====================== test ======================
        # Model.test(test_loader,num)
        logging.info(f'the {num} turn finished!!!!!!!!!!')

    save_name = file_path + "_final_results.csv"
    files_path = file_path + "*.txt"
    cal_ave_result(files_path, save_name)