import numpy as np
from scipy import stats

import torch
from torch.autograd import Variable

import os
import shutil

from .. import cde
from ..utils import systematic_resample, ESS


RESAMPLE_THRESH = 0.5
SIGMA = 20.0
P_INIT = 0.1
TRANSITION = np.array([[0.95, 0.05],
                       [0.05, 0.95]])


def gen_devices(num_devices=20):
    """ Produce a synthetic set of "devices", i.e. a set of mean parameters for different
        hypothetical applicances """
    np.random.seed(0)
    return np.random.permutation(np.array(np.round(np.linspace(3,50,num_devices)), dtype=int)*10)


def gen_dataset(devices, T=200):
    """ Sample a time series of synthetic data for a given set of devices """
    devices = np.array(devices, dtype=float)
    D = len(devices)
    X = np.zeros((T, D), dtype=int)
    Y = np.zeros((T,))
    sample_additive = lambda x: max(0, np.dot(x, devices) + np.random.randn()*SIGMA)
    X[0] = np.random.rand(D) < P_INIT
    Y[0] = sample_additive(X[0])
    for t in xrange(1,T):
        X[t] = (TRANSITION[X[t-1]].cumsum(1) < np.random.rand(D,1)).sum(1)
        Y[t] = sample_additive(X[t])
        
    return X, Y


def get_training_data(batch_size, devices):
    """ return synthetic training data (inputs, outputs) """
    D = len(devices)
    T = 51
    batch_size /= (T-1)
    data = np.empty((batch_size*(T-1), 2*D+1))
    for i in xrange(batch_size):
        X, Y = gen_dataset(devices, T)
        inputs = np.hstack((X[:-1],Y[1:,None]))
        outputs = X[1:]
        data[i*(T-1):(i+1)*(T-1),:] = np.hstack((inputs, outputs))
    return data[:,:D+1], data[:,D+1:]
    


def _iterate_minibatches(inputs, outputs, batch_size):
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        excerpt = slice(start_idx, start_idx + batch_size)
        yield Variable(torch.FloatTensor(inputs[excerpt])), Variable(torch.FloatTensor(outputs[excerpt]))

def training_step(optimizer, dist_est, devices, dataset_size, batch_size, max_local_iters=10, misstep_tolerance=0, verbose=False):
    """ Training function for fitting density estimator to simulator output """
    # Train
    USE_GPU = dist_est.parameters().next().is_cuda
    synthetic_ins, synthetic_outs = get_training_data(dataset_size, devices)
    ordering = np.random.permutation(synthetic_ins.shape[0])
    synthetic_ins = synthetic_ins[ordering]
    synthetic_outs = synthetic_outs[ordering]
    if max_local_iters > 1:
        validation_size = dataset_size/10
        validation_ins, validation_outs = [Variable(torch.FloatTensor(v)) for v in get_training_data(validation_size, devices)]
        if USE_GPU:
            validation_ins = validation_ins.cuda()
            validation_outs = validation_outs.cuda()
        validation_err = -torch.mean(dist_est.logpdf(validation_ins, validation_outs)).data[0]
    else:
        validation_err = None

    missteps = 0
    num_batches = float(synthetic_ins.shape[0])/batch_size

    for local_iter in xrange(max_local_iters):
    
        train_err = 0 
        for inputs, outputs in _iterate_minibatches(synthetic_ins, synthetic_outs, batch_size):
            optimizer.zero_grad()
            if USE_GPU:
                loss = -torch.mean(dist_est.logpdf(inputs.cuda(), outputs.cuda()))
            else:
                loss = -torch.mean(dist_est.logpdf(inputs, outputs))
            loss.backward()
            params_before = [np.isnan(p.data.numpy()).sum() for p in dist_est.parameters()]
            assert sum(params_before) == 0
            optimizer.step()
            params_after = [np.isnan(p.data.numpy()).sum() for p in dist_est.parameters()]
            assert sum(params_after) == 0
            train_err += loss.data[0]/num_batches
            
        if max_local_iters > 1:
            next_validation_err = -torch.mean(dist_est.logpdf(validation_ins, validation_outs)).data[0]
            if next_validation_err > validation_err:
                missteps += 1
            validation_err = next_validation_err
            if missteps > misstep_tolerance:
                break

    if verbose:
        print train_err, validation_err, "(", local_iter+1, ")"
    return train_err, validation_err, local_iter+1
    

def baseline_proposal(x, y):
    next_x = np.zeros_like(x)
    K, D = x.shape
    for k in xrange(K):
        next_x[k] = (TRANSITION[x[k]].cumsum(1) < np.random.rand(D,1)).sum(1)
    ln_q = np.sum(np.log(TRANSITION[0,0]) * (next_x == x) + 
                  np.log(TRANSITION[0,1]) * (next_x != x), 1)
    return next_x, ln_q

def make_nn_proposal(dist_est):
    USE_GPU = dist_est.parameters().next().is_cuda
    def nn_proposal(x, y):
        K, D = x.shape
        state = np.concatenate((x,y*np.ones((K,1))), axis=1)
        if USE_GPU:
            state = Variable(torch.cuda.FloatTensor(state))
        else:
            state = Variable(torch.FloatTensor(state))
        val, ln_q = dist_est.propose(state)
        return val.data.cpu().numpy(), ln_q.data.cpu().numpy().squeeze()
    return nn_proposal

def run_smc(devices, Y, K, proposal, verbose=True):
    """ Run an SMC algorithm using K particles, and proposal distribution `proposal`,
        which returns a value and its proposal log probability.
        
        `factorial_hmm.baseline_proposal` samples from the transition dynamics.
        `factorial_hmm.make_nn_proposal` generates a proposal using a learned network. """
        
    T = len(Y)
    X_hat = np.zeros((K,T,len(devices)), dtype=int)
    ancestry = np.empty((K,T), dtype=int)
    ln_q = 0.0
    X_hat[:,0], ln_q = proposal(1*(np.random.rand(K, len(devices)) < P_INIT), Y[0])
    log_weights = stats.norm(Y[0], SIGMA).logpdf(np.dot(X_hat[:,0], devices)) - ln_q
    ESS_history = np.empty((T,))
    ESS_history[0] = ESS(log_weights)
    if ESS_history[0] < K*RESAMPLE_THRESH:
        X_hat = X_hat[systematic_resample(log_weights)]
        log_weights[:] = 0.0 # np.log(np.mean(np.exp(log_weights)))
    ancestry[:,0] = np.arange(K)
    for t in xrange(1,len(Y)):
        X_hat[:,t], ln_q = proposal(X_hat[:,t-1], Y[t])
        ln_p_trans = np.sum(np.log(TRANSITION[0,0]) * (X_hat[:,t] == X_hat[:,t-1]) + 
                            np.log(TRANSITION[0,1]) * (X_hat[:,t] != X_hat[:,t-1]), 1)
        # assert np.isfinite(ln_q) # TODO add back
        log_weights += stats.norm(Y[t], SIGMA).logpdf(np.dot(X_hat[:,t], devices)) + ln_p_trans - ln_q
        ESS_history[t] = ESS(log_weights)
        if ESS_history[t] < K*RESAMPLE_THRESH:
            if verbose:
                print "RESAMPLE", t, ESS_history[t]
            indices = systematic_resample(log_weights)
            X_hat = X_hat[indices]
            log_weights[:] = 0.0 # np.log(np.mean(np.exp(log_weights)))
            ancestry = ancestry[indices]
        ancestry[:,t] = np.arange(K)
    indices = systematic_resample(log_weights)
    ancestry = ancestry[indices]
    ancestry[:,-1] = np.arange(K)
    return X_hat[indices], ancestry, ESS_history
    
if __name__ == '__main__':
    USE_GPU = torch.cuda.is_available()
    print "Using GPU?", USE_GPU
    devices = gen_devices()
    trace_train = []
    trace_validation = []

    dist_est = cde.ConditionalBinaryMADE(len(devices)+1, len(devices), H=300, num_layers=4)
    if USE_GPU:
        dist_est.cuda()
    optimizer = torch.optim.Adam(dist_est.parameters(), lr=0.001)
    num_iterations = 2000
    dataset_size = 5000
    batch_size = 500
    
    outfile = 'trained_hmm_params.rar'
    if os.path.exists(outfile):
        shutil.copyfile(outfile, '{}.backup'.format(outfile))

    for i in xrange(num_iterations):
        verbose = True
        print "["+str(1+len(trace_train))+"]",
        t,v,_ = training_step(optimizer, dist_est, devices, dataset_size, batch_size, max_local_iters=10, verbose=True)
        trace_train.append(t)
        trace_validation.append(v)
        torch.save(dist_est.cpu().state_dict(), outfile)
        if USE_GPU: dist_est.cuda()
