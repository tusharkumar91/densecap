import torch

class MaxMarginCriterion(torch.nn.Module):
    def __init__(self, margin):
        super(MaxMarginCriterion, self).__init__()
        self.margin = margin

    def forward(self, cossim, target):
        N = cossim.size(0)
        choices = cossim.size(1)
        loss = 0
        for batch in range(N):
            correct_idx = (target[batch] == 1).nonzero()[0]
            correct_sim = cossim[batch][correct_idx]
            batch_loss = 0
            count = 0
            for ch_idx in range(choices):
                if ch_idx != correct_idx:
                    count += 1
                    batch_loss += torch.clamp(self.margin + cossim[batch][ch_idx] - correct_sim, min=0)
            #batch_loss = batch_loss / count
            loss += batch_loss
        return loss / N
