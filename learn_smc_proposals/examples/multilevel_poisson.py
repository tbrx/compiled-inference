import numpy as np
import torch
from scipy import stats
from torch.autograd import Variable

import pymc
import os
import shutil

from learn_smc_proposals import cde

num_points = 10 # number of points in the synthetic dataset we train on

def gamma_poisson(x, t):
    """ x: number of failures (N vector)
        t: operation time, thousands of hours (N vector) """
    
    if x is not None:
        N = x.shape
    else:
        N = num_points
    
    # place an exponential prior on t, for when it is unknown
    t = pymc.Exponential('t', beta=1.0/50.0, value=t, size=N, observed=(t is not None))
    
    alpha = pymc.Exponential('alpha', beta=1.0, value=1.0)
    beta = pymc.Gamma('beta', alpha=0.1, beta=1.0, value=1.0)
    
    theta = pymc.Gamma('theta', alpha=alpha, beta=beta, size=N)
    
    @pymc.deterministic
    def mu(theta=theta, t=t):
        return theta*t

    x = pymc.Poisson('x', mu=mu, value=x, observed=(x is not None))
    
    return locals()

sample_test_points = lambda: num_points

def get_input_theta(model):
    return np.atleast_2d(np.vstack((model.x.value[:1], model.t.value[:1])))
#     return np.atleast_2d(np.vstack((model.x.value, model.t.value)))

def get_target_theta(model):
    return np.atleast_2d(model.theta.value[:1])
#     return np.atleast_2d(model.theta.value)

#get_input_params = get_target_theta


def get_input_params(model):
    return np.atleast_2d(model.theta.value)

def get_target_params(model):
    return np.atleast_2d([model.alpha.value, model.beta.value])


def generate_synthetic(model, size=100):
    # N = sample_test_points()
    ins_theta, outs_theta = None, None
    ins_params, outs_params = None, None
    #while len(ins_params) < size:
    for i in xrange(size-1):
        try:
            model.draw_from_prior()
            if np.min(get_target_params(model).ravel()) < 1e-5 or \
               np.max(get_target_params(model).ravel()) > 1e5 or \
               np.min(get_input_params(model).ravel()) < 1e-5 or \
               np.max(get_input_params(model).ravel()) > 1e5:
                # filter out garbage
                continue
            if ins_theta is None:
                ins_theta, outs_theta = get_input_theta(model).T, get_target_theta(model).T
                ins_params, outs_params = get_input_params(model), get_target_params(model)
            else:
                ins_theta = np.vstack((ins_theta, get_input_theta(model).T))
                outs_theta = np.vstack((outs_theta, get_target_theta(model).T))
                ins_params = np.vstack((ins_params, get_input_params(model)))
                outs_params = np.vstack((outs_params, get_target_params(model)))
        except Exception as e:
            #print e
            pass
    theta = (ins_theta, outs_theta)
    params = (ins_params, outs_params)
    #theta = (np.log(ins_theta), np.log(outs_theta))
    #params = (np.log(ins_params), np.log(outs_params))
    return theta, params

def _iterate_minibatches(inputs, outputs, batchsize):
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        excerpt = slice(start_idx, start_idx + batchsize)
        yield Variable(torch.FloatTensor(inputs[excerpt])), Variable(torch.FloatTensor(outputs[excerpt]))

def training_step(target_model, batch_size, max_local_iters, misstep_tolerance=0, verbose=False, generator=generate_synthetic):
    """ Training function for fitting density estimator to simulator output """
    # Train
    synthetic_data = generator(M_train, batch_size*10)[target_model]
    dataset_size = synthetic_data[0].shape[0]
    print ("(size=%d):\t" % dataset_size),
    validation_size = dataset_size/10
    validation_data = [Variable(torch.FloatTensor(t)) for t in generator(M_train, validation_size)[target_model]]
    USE_GPU = estimators[target_model].parameters().next().is_cuda
    if USE_GPU:
        validation_data = [d.cuda() for d in validation_data]

    missteps = 0
    num_batches = float(dataset_size)/batch_size
    
#     backup = estimators[target_model].state_dict()
    validation_err = -estimators[target_model].logpdf(validation_data[0], validation_data[1]).mean()
    validation_err = validation_err.data[0]
    for local_iter in xrange(max_local_iters):
        train_err = 0 
        for inputs, outputs in _iterate_minibatches(synthetic_data[0], synthetic_data[1], batch_size):
            optimizers[target_model].zero_grad()
            if USE_GPU:
                loss = -torch.mean(estimators[target_model].logpdf(inputs.cuda(), outputs.cuda()))
            else:
                loss = -torch.mean(estimators[target_model].logpdf(inputs, outputs))
            loss.backward()
            optimizers[target_model].step()
            train_err += loss.data[0]/num_batches

            
        next_validation_err = -estimators[target_model].logpdf(*validation_data).mean()
#         if np.isnan(train_err) or np.isnan(next_validation_err.data[0]):
#             estimators[target_model].load_state_dict(backup)
#             break
        if next_validation_err > validation_err:
            missteps += 1
        validation_err = next_validation_err.data[0]
        if missteps > misstep_tolerance:
            break

    if verbose:
        print round(train_err, 4), round(validation_err, 4), "(", local_iter+1, ")",
        
    return train_err, validation_err, local_iter+1

def get_estimators():
    theta_est = cde.ConditionalRealValueMADE(2, 1, 500, 2, 10)
    params_est = cde.ConditionalRealValueMADE(10, 2, 500, 2, 10)
    return theta_est, params_est


if __name__ == '__main__':
    M_train = pymc.Model(gamma_poisson(None, None))
    theta_est, params_est = get_estimators()
    USE_GPU = torch.cuda.is_available()
    if USE_GPU:
        theta_est.cuda()
        params_est.cuda()
    estimators = [theta_est, params_est]
    optimizers = [torch.optim.Adam(model.parameters()) for model in estimators]

    trace_train = ([], [])
    trace_validation = ([], [])
    trace_local_iters = ([], [])

    num_steps = 500
    batch_size = 2500
    max_local_iters = 10

    file1 = 'trained_poisson_theta.rar'
    file2 = 'trained_poisson_params.rar'
    if os.path.exists(file1):
        shutil.copyfile(file1, '{}.backup'.format(file1))
    if os.path.exists(file2):
        shutil.copyfile(file2, '{}.backup'.format(file2))

    for i in xrange(num_steps):
        for target_model in [0,1]:
            verbose = True #  (i+1) % 1 == 0
            if verbose:
                print "["+("thetas","params")[target_model]+" "+str(1+len(trace_train[target_model]))+"]",
            t,v,l = training_step(target_model, batch_size, max_local_iters, verbose=verbose)
            trace_train[target_model].append(t)
            trace_validation[target_model].append(v)
            if verbose: print '\t',
            trace_local_iters[target_model].append(l)
        if verbose:
            print
        torch.save(theta_est.cpu().state_dict(), file1)
        torch.save(params_est.cpu().state_dict(), file2)
        if USE_GPU:
            theta_est.cuda()
            params_est.cuda()
        
