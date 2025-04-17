import torch
import pickle
from torch.utils.data import Dataset
from parameter_setting import parse_args
args = parse_args()

class Mydata(Dataset):
    def __init__(self, city, pad_num, max_seq, data_part, device):
        super(Mydata, self).__init__()
        self.u = []
        self.s = []
        self.l = []
        self.c = []
        self.h = []
        self.w = []
        with open(f'./data/{city}_data_{data_part}_{args.graph}.pkl', 'rb') as file:
            data_set = pickle.load(file) 
        # len = 15, [[user],[input_s],[train_gt],[cat_s],[cat_gt],[lat_s],[lat_gt],[long_s],[long_gt],[hour_s],[hour_gt],[weekday_s],[weekday_gt],[cat1_s],[cat1_gt]]
        for seq in data_set[1]:
            if len(seq) < max_seq:
                padding = [pad_num]*(max_seq-len(seq))
                padding.extend(seq)
                self.s.append(torch.tensor(padding).to(device))
            else:
                self.s.append(torch.tensor(seq).to(device))
        for user in data_set[0]:
            self.u.append(torch.tensor(user).to(device))
        for label in data_set[2]:
            self.l.append(torch.tensor(label).to(device))
        for cat in data_set[13]:
            if len(cat) < max_seq:
                padding = [pad_num]*(max_seq-len(cat))
                padding.extend(cat)
                self.c.append(torch.tensor(padding).to(device))
            else:
                self.c.append(torch.tensor(cat).to(device))
        for hour in data_set[9]:
            if len(hour) < max_seq:
                padding = [pad_num]*(max_seq-len(hour))
                padding.extend(hour)
                self.h.append(torch.tensor(padding).to(device))
            else:
                self.h.append(torch.tensor(hour).to(device))
        for weekday in data_set[11]:
            if len(weekday) < max_seq:
                padding = [pad_num]*(max_seq-len(weekday))
                padding.extend(weekday)
                self.w.append(torch.tensor(padding).to(device))
            else:
                self.w.append(torch.tensor(weekday).to(device))
        self.x = list(zip(self.u,self.s,self.l,self.c,self.h,self.w))
    
    def __getitem__(self, idx):
        user = self.u[idx]
        seq = self.s[idx]
        label = self.l[idx]
        cat = self.c[idx]
        hour = self.h[idx]
        weekday = self.w[idx]
        return user, seq, label, cat, hour, weekday

    def __len__(self):
        return len(self.x)