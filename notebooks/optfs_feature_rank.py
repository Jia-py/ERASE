import numpy as np
import torch

path = 'checkpoints/mlp_optfs_no_selection_aliccp_1704885845.1920161/model_search.pth'
dataset = 'aliccp'

model = torch.load(path)
features = []
importance = []

if dataset == 'movielens-1m':
    features = ['user_id', 'movie_id', 'timestamp', 'title', 'genres', 'gender', 'age', 'occupation', 'zip']
    unique_values = [6040, 3706, 458455, 3706, 301, 2, 7, 21, 3439]
    offsets = np.array((0, *np.cumsum(unique_values)[:-1], np.sum(unique_values)))
elif dataset == 'aliccp':
    features = ['101', '121', '122', '124', '125', '126', '127', '128', '129', '205', '206', '207', '210', '216', '508', '509', '702', '853', '301', '109_14', '110_14', '127_14', '150_14']
    unique_values = [238635, 98, 14, 3, 8, 4, 4, 3, 5, 467298, 6929, 263942, 80232, 106399, 5888, 104830, 51878, 37148, 3, 5853, 105622, 53843, 31858]
    offsets = np.array((0, *np.cumsum(unique_values)[:-1], np.sum(unique_values)))

for name, param in model.items():
    if 'fs.mask_weight' in name:
        tmp = param.detach().cpu().numpy().reshape(-1,1)
        tmp[tmp<=0]=0
        tmp[tmp>0]=1
        for i, feature in enumerate(features):
            start_idx = offsets[i]
            end_idx = offsets[i+1]
            print(feature)
            print(np.sum(tmp[start_idx:end_idx]))
            print(tmp[start_idx:end_idx].shape[0])
            importance.append(np.sum(tmp[start_idx:end_idx]) / tmp[start_idx:end_idx].shape[0])

rank = np.argsort(importance)[::-1]
ranked_features = np.array(features)[rank]
ranked_importance = np.array(importance)[rank]
print(ranked_features)
print(ranked_importance)
np.save('optfs_rank.npy',np.array([ranked_features, ranked_importance]))


