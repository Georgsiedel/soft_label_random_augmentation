import torch
import torch.nn.functional as F

def soft_loss(pred, label, confidence, reweight=False):
    #see https://arxiv.org/pdf/2211.04625 and https://github.com/youngleox/soft_augmentation for reference
    
    log_prob = F.log_softmax(pred, dim=1)
    n_class = pred.size(1)

    # Make soft one-hot target with correct label = confidence and the inconfidence uniformly distributed
    label = label.unsqueeze(1)
    confidence = confidence.unsqueeze(1).float()

    one_hot = torch.ones_like(pred) * (1 - confidence) / (n_class - 1)
    one_hot.scatter_(dim=1, index=label, src=confidence)
    
    # Compute weighted KL loss
    kl = F.kl_div(input=log_prob, target=one_hot, reduction="none").sum(-1)
    kl = kl.unsqueeze(1)  # Unweighted
    if reweight:
        kl = confidence * kl  # Weighted
    return kl.mean()