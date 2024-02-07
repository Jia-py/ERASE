import torch
import torch.nn as nn
import numpy as np

class no_selection(nn.Module):
    def __init__(self):
        super(no_selection, self).__init__()

    def forward(self, x):
        return x

# class no_selection(nn.Module):
#     def __init__(self,args,unique_values, features):
#         super(no_selection, self).__init__()
#         self.feature_num = len(unique_values)
#         self.device = args.device
#         self.args = args
#         self.features = np.array(features)
#         self.embed_dim = args.embedding_dim
#         self.offsets = np.array((0, *np.cumsum(unique_values)[:-1]))
#         self.embedding = nn.Embedding(sum(unique_values), embedding_dim = self.embed_dim)
#         torch.nn.init.normal_(self.embedding.weight.data, mean=0, std=0.01)

#     def forward(self, x):
#         x = self.embedding(x + x.new_tensor(self.offsets))
#         return x