import numpy as np
import torch
import torch.nn.functional as F

mine = torch.from_numpy(np.load('/tmp/my_hidden_traj.npy')).float()
official = torch.from_numpy(np.load('/tmp/official_hidden_traj.npy')).float()
print({'shapes': [list(mine.shape), list(official.shape)]}, flush=True)

frames = min(len(mine), len(official))
cos = [round(float(F.cosine_similarity(mine[i], official[i], dim=0)), 3) for i in range(frames)]
print({'per_frame_cos': cos[:12]}, flush=True)
print({'cos_mean': round(sum(cos) / len(cos), 3), 'cos_last': cos[-1]}, flush=True)

def step_size(traj):
    diffs = (traj[1:] - traj[:-1]).norm(dim=-1)
    return round(float(diffs.mean()), 2), round(float(diffs.max()), 2)

print({'official_step': step_size(official), 'mine_step': step_size(mine)}, flush=True)
print({'norms': {'official': round(float(official.norm(dim=-1).mean()), 2),
                  'mine': round(float(mine.norm(dim=-1).mean()), 2)}}, flush=True)