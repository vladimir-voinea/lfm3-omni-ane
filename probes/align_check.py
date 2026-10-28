import numpy as np
import torch
import torch.nn.functional as F

mine = torch.from_numpy(np.load('/tmp/my_hidden_traj.npy')).float()
official = torch.from_numpy(np.load('/tmp/official_hidden_traj.npy')).float()
standalone = torch.from_numpy(np.load('/tmp/official_frame0_hidden.npy')).float().reshape(-1)

print('mine[0] vs standalone official frame0:', round(float(F.cosine_similarity(mine[0], standalone, dim=0)), 4), flush=True)
for j in range(4):
    print('mine[0] vs official_traj[%d]:' % j,
          round(float(F.cosine_similarity(mine[0], official[j], dim=0)), 4), flush=True)
print('official_traj[0] vs standalone:', round(float(F.cosine_similarity(official[0], standalone, dim=0)), 4), flush=True)