import numpy as np

def systematic_resample(log_weights):
    A = log_weights.max()
    normalizer = np.log(np.exp(log_weights - A).sum()) + A
    weights = np.exp(log_weights - normalizer)
    ns = len(weights)
    cdf = np.cumsum(weights)
    cutoff = (np.random.rand() + np.arange(ns))/ns
    return np.digitize(cutoff, cdf)

def ESS(log_weights):
    A = log_weights.max()
    normalizer = np.log(np.exp(log_weights - A).sum()) + A
    log_normalized = 2*(log_weights - normalizer)
    B = log_normalized.max()
    log_denominator = np.log(np.sum(np.exp(log_normalized - B))) + B
    return np.exp(-log_denominator)

