import torch
import torch.nn.functional as F

def contrastive_loss(z1, z2, temperature=0.5):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    batch_size = z1.size(0)

    representations = torch.cat([z1, z2], dim=0)
    similarity_matrix = torch.matmul(representations, representations.T)

    mask = torch.eye(2 * batch_size, device=z1.device).bool()
    similarity_matrix = similarity_matrix[~mask].view(2 * batch_size, -1)

    positives = torch.cat([
        torch.diag(torch.matmul(z1, z2.T)),
        torch.diag(torch.matmul(z2, z1.T))
    ], dim=0)

    positives = positives.unsqueeze(1)

    logits = torch.cat([positives, similarity_matrix], dim=1)
    labels = torch.zeros(2 * batch_size, dtype=torch.long).to(z1.device)

    logits = logits / temperature

    return F.cross_entropy(logits, labels)