"""Small residual U-Net guided by a registered appearance candidate."""
import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class RemovalNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.shallow = nn.Sequential(nn.Conv2d(8, 16, 3, padding=1), nn.ReLU())
        self.deep = nn.Sequential(nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
                                  nn.Conv2d(32, 32, 3, padding=1), nn.ReLU())
        self.merge = nn.Sequential(nn.Conv2d(48, 16, 3, padding=1), nn.ReLU(), nn.Conv2d(16, 3, 1))
        nn.init.zeros_(self.merge[-1].weight)
        nn.init.zeros_(self.merge[-1].bias)

    def forward(self, x):
        skip = self.shallow(x)
        low = F.interpolate(self.deep(skip), size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return x[:,4:7] + self.merge(torch.cat((skip, low), dim=1))


def inputs(images, edit, candidate, known):
    raw = np.where(edit[...,None], 0, images).astype(np.float32)/255
    planes = np.concatenate((raw, edit[...,None], candidate.astype(np.float32)/255, known[...,None]), axis=-1)
    return torch.from_numpy(planes).permute(0,3,1,2).float()


def objective(pred, target, edit):
    delta = pred-target
    def weighted(value, mask):
        return (value.abs()*mask).sum()/(3*mask.sum().clamp_min(1))
    return (weighted(delta, edit)
            + .2*weighted(delta.diff(dim=2), edit[:,:,1:] * edit[:,:,:-1])
            + .2*weighted(delta.diff(dim=3), edit[:,:,:,1:] * edit[:,:,:,:-1]))


def temporal_pairs(target):
    """Backward sampling grids on clean training pairs with a round-trip visibility test."""
    grays = [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
             for im in target.permute(0,2,3,1).mul(255).byte().numpy()]
    h, w = grays[0].shape
    y,x = np.indices((h,w), dtype=np.float32)
    points = np.stack((x,y), axis=-1)
    grids, valid = [], []
    for a,b in zip(grays,grays[1:]):
        fwd = cv2.calcOpticalFlowFarneback(a,b,None,.5,3,15,3,5,1.2,0)
        back = cv2.calcOpticalFlowFarneback(b,a,None,.5,3,15,3,5,1.2,0)
        donor = points + back
        error = np.linalg.norm(back + cv2.remap(fwd, donor, None, cv2.INTER_LINEAR), axis=-1)
        photo = np.abs(b.astype(float)-cv2.remap(a, donor, None, cv2.INTER_LINEAR))
        inside = ((donor[...,0]>=0) & (donor[...,0]<w-1) &
                  (donor[...,1]>=0) & (donor[...,1]<h-1))
        valid.append(inside & (error<1) & (photo<10))
        grids.append(2*donor / np.float32([w-1,h-1])-1)
    return torch.from_numpy(np.stack(grids)), torch.from_numpy(np.stack(valid))[:,None]


def fit(training, validation, directory, epochs):
    torch.manual_seed(41)
    model = RemovalNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    paired = [(x,y,*temporal_pairs(y)) for x,y in training if len(y)>1]
    history, best = [], np.inf
    for epoch in range(epochs):
        losses = []
        for x,y in training:
            for ids in torch.randperm(len(x)).split(4):
                optimizer.zero_grad()
                loss = objective(model(x[ids]), y[ids], x[ids,3:4])
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
        for x,y,grids,visible in paired:
            for t in torch.randperm(len(grids)).split(4):
                earlier = model(x[t])-y[t]
                later = model(x[t+1])-y[t+1]
                warped = F.grid_sample(earlier, grids[t], align_corners=True)
                use = visible[t] * x[t+1,3:4]
                temporal = ((later-warped).abs()*use).sum()/(3*use.sum().clamp_min(1))
                optimizer.zero_grad()
                (.1*temporal).backward()
                optimizer.step()
        with torch.no_grad():
            score = np.mean([objective(model(x), y, x[:,3:4]).item() for x,y in validation])
        history.append(dict(epoch=epoch+1, training=float(np.mean(losses)), validation=float(score)))
        if score < best:
            best = score
            torch.save(model.state_dict(), directory/"weights.pt")
    return history
