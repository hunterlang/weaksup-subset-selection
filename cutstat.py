import torch
def get_cutstat_subset(dataset, features, labels, coverage=0.5, K=20, device='cpu'):
    features = torch.FloatTensor(features).to(device)
    labels = torch.LongTensor(labels).to(device)
    with torch.no_grad():
        keep_indices = get_cutstat_inds(features,
                                        labels,
                                        coverage=coverage,
                                        device=device)

        # wrench dataset has create_subset() method that takes indices
        dataset = dataset.create_subset(keep_indices)
        labels = labels[keep_indices]

    return dataset, labels


def get_cutstat_inds(features, labels, coverage=0.5, K=20, device='cpu'):
        # move to CPU for memory issues on large dset
        pairwise_dists = torch.cdist(features, features, p=2).to('cpu')

        N = labels.shape[0]
        dists_sorted = torch.argsort(pairwise_dists)
        neighbors = dists_sorted[:,:K]
        dists_nn = pairwise_dists[torch.arange(N)[:,None], neighbors]
        weights = 1/(1 + dists_nn)
        neighbors = neighbors.to(device)
        dists_nn = dists_nn.to(device)
        weights = weights.to(device)
        cut_vals = (labels[:,None] != labels[None,:]).long()
        cut_neighbors = cut_vals[torch.arange(N)[:,None], neighbors]
        Jp = (weights * cut_neighbors).sum(dim=1)
        weak_counts = torch.bincount(labels)
        weak_pct = weak_counts / weak_counts.sum()
        prior_probs = weak_pct[labels]
        mu_vals = (1-prior_probs) * weights.sum(dim=1)
        sigma_vals = prior_probs * (1-prior_probs) * torch.pow(weights, 2).sum(dim=1)
        sigma_vals = torch.sqrt(sigma_vals)
        normalized = (Jp - mu_vals) / sigma_vals
        normalized = normalized.cpu()
        inds_sorted = torch.argsort(normalized)
        N_select = int(coverage * N)
        conf_inds = inds_sorted[:N_select]
        conf_inds = list(set(conf_inds.tolist()))
        return conf_inds
