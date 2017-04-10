import torch
import torch.nn as nn
import torch.tensor as tensor
import torch.nn.init as init
from torch.autograd import Variable

import numpy as np

def sample_mask_indices(D, H, simple=False):
    if simple:
        return np.random.randint(0, D, size=(H,))
    else:
        mk = np.linspace(0,D-1,H)
        ints = np.array(mk, dtype=int)
        ints += (np.random.rand() < mk - ints)
        return ints

def create_mask(D_observed, D_latent, H, num_layers):
    m_input = np.concatenate((np.zeros(D_observed), 1+np.arange(D_latent)))
    m_w = [sample_mask_indices(D_latent, H) for i in range(num_layers)]
    m_v = np.arange(D_latent)
    M_A = (1.0*(np.atleast_2d(m_v).T >= np.atleast_2d(m_input)))
    M_W = [(1.0*(np.atleast_2d(m_w[0]).T >= np.atleast_2d(m_input)))]
    for i in range(1, num_layers):
        M_W.append(1.0*(np.atleast_2d(m_w[i]).T >= np.atleast_2d(m_w[i-1])))
    M_V = (1.0*(np.atleast_2d(m_v).T >= np.atleast_2d(m_w[-1])))

    return M_W, M_V, M_A


class MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, mask, bias=True):
        """ Linear layer, but with a mask. 
            Mask should be a tensor of size (out_features, in_features). """
        super(MaskedLinear, self).__init__(in_features, out_features, bias)
        self.register_buffer('mask', mask)

    def forward(self, input):
        mask = Variable(self.mask, requires_grad=False)
        if self.bias is None:
            return self._backend.Linear()(input, self.weight*mask)
        else:
            return self._backend.Linear()(input, self.weight*mask, self.bias)


class AbstractConditionalMADE(nn.Module):
    def __init__(self, D_observed, D_latent, H, num_layers):
        super(AbstractConditionalMADE, self).__init__()
        self.D_in = D_observed + D_latent
        self.D_out = D_latent
        assert num_layers >= 1
    
        # create masks
        M_W, M_V, M_A = create_mask(D_observed, D_latent, H, num_layers)
        self.M_W = [torch.FloatTensor(M) for M in M_W]
        self.M_V = torch.FloatTensor(M_V)
        self.M_A = torch.FloatTensor(M_A)

        # nonlinearities
        self.relu = nn.ReLU()

    def forward(self, x):
        raise NotImplementedError()
        
    def sample(self, parents):
        raise NotImplementedError()
        
    def logpdf(self, parents, values):
        raise NotImplementedError()

    def propose(self, parents, ns=1):
        """ Given a setting of the observed (parent) random variables, sample values of 
            the latents; returns a tuple of both the values and the log probability under 
            the proposal.
        
            If ns > 1, each of the tensors has an added first dimension of ns, each 
            containing a sample of size [batch_size, D_latent] and [batch_size] """
        original_batch_size = parents.size(0)
        if ns > 1:
            parents = parents.repeat(ns,1)
        values = self.sample(parents)
        ln_q = self.logpdf(parents, values)        
        if ns > 1:
            values = values.resize(ns, original_batch_size, self.D_out)
            ln_q = ln_q.resize(ns, original_batch_size)
        return values, ln_q


class ConditionalBinaryMADE(AbstractConditionalMADE):
    def __init__(self, D_observed, D_latent, H, num_layers):
        super(ConditionalBinaryMADE, self).__init__(D_observed, D_latent, H, num_layers)
        
        # layers
        layers = [MaskedLinear(self.D_in, H, self.M_W[0])]
        for i in xrange(1,num_layers):
            layers.append(MaskedLinear(H, H, self.M_W[i]))
        self.layers = nn.ModuleList(layers)
        self.skip_p = MaskedLinear(self.D_in, self.D_out, self.M_A, bias=False)
        self.skip_q = MaskedLinear(self.D_in, self.D_out, self.M_A, bias=False)
        self.p = MaskedLinear(H, self.D_out, self.M_V)
        self.q = MaskedLinear(H, self.D_out, self.M_V)
        self.loss = nn.BCELoss(size_average=True)
        
        # initialize parameters
        for param in self.parameters():
            if len(param.size()) == 1:
                init.normal(param, std=0.01)
            else:
                init.uniform(param, a=-0.01, b=0.01)

    def forward(self, x):
        h = x
        for i, layer in enumerate(self.layers):
            h = self.relu(layer(h))
        epsilon = 1e-8
        logp = self.p(h) + self.skip_p(x) + epsilon
        logq = self.q(h) + self.skip_q(x) + epsilon
        A = torch.max(logp, logq)
        normalizer = ((logp - A).exp_() + (logq - A).exp_()).log() + A
#         assert (1 - np.isfinite(normalizer.data.numpy())).sum() == 0
        logp -= normalizer
        return logp.exp_()

    def sample(self, parents):
        """ Given a setting of the observed (parent) random variables, sample values of the latents """
        assert parents.size(1) == self.D_in - self.D_out
        batch_size = parents.size(0)
        FloatTensor = torch.cuda.FloatTensor if parents.is_cuda else torch.FloatTensor
        latent = Variable(FloatTensor(batch_size, self.D_out))
        randvals = Variable(FloatTensor(batch_size, self.D_out))
        torch.rand(batch_size, self.D_out, out=randvals.data)
        for d in xrange(self.D_out):
            full_input = torch.cat((parents, latent), 1)
            latent = torch.ge(self(full_input), randvals).float()
        return latent
    
    def logpdf(self, parents, values):
        """ Return the conditional log probability `ln p(values|parents)` """
        p = self.forward(torch.cat((parents, values), 1))
        p = torch.clamp(p, 1e-6, 1.0 - 1e-6)
        return -self.loss(p, values)*values.size(1)



class ConditionalRealValueMADE(AbstractConditionalMADE):
    def __init__(self, D_observed, D_latent, H, num_layers, num_components):
        super(ConditionalRealValueMADE, self).__init__(D_observed, D_latent, H, num_layers)
        
        self.K = num_components
        self.softplus = nn.Softplus()
        
        layers = [MaskedLinear(self.D_in, H, self.M_W[0])]
        for i in xrange(1,num_layers):
            layers.append(MaskedLinear(H, H, self.M_W[i]))
        self.layers = nn.ModuleList(layers)
        skip_alpha,skip_mu,skip_sigma,mu,alpha,sigma = [],[],[],[],[],[]
        for k in xrange(num_components):
            skip_alpha.append(MaskedLinear(self.D_in, self.D_out, self.M_A, bias=False))
            skip_mu.append(MaskedLinear(self.D_in, self.D_out, self.M_A, bias=False))
            skip_sigma.append(MaskedLinear(self.D_in, self.D_out, self.M_A, bias=False))
            alpha.append(MaskedLinear(H, self.D_out, self.M_V))
            mu.append(MaskedLinear(H, self.D_out, self.M_V))
            sigma.append(MaskedLinear(H, self.D_out, self.M_V))
        self.skip_alpha = nn.ModuleList(skip_alpha)
        self.skip_mu = nn.ModuleList(skip_mu)
        self.skip_sigma = nn.ModuleList(skip_sigma)
        self.alpha = nn.ModuleList(alpha)
        self.mu = nn.ModuleList(mu)
        self.sigma = nn.ModuleList(sigma)
    
        # initialize parameters
        for param in self.parameters():
            if len(param.size()) == 1:
                init.normal(param, std=0.01)
            else:
                init.uniform(param, a=-0.01, b=0.01)

    def forward(self, x):
        h = x
        for i, layer in enumerate(self.layers):
            h = self.relu(layer(h))
        log_alpha = [self.alpha[k](h) + self.skip_alpha[k](x) for k in xrange(self.K)]
        mu = [self.mu[k](h) + self.skip_mu[k](x) for k in xrange(self.K)]
        sigma = [self.softplus(self.sigma[k](h) + self.skip_sigma[k](x)) for k in xrange(self.K)]
        
        alpha = torch.cat(map(lambda x: x.unsqueeze(2), log_alpha), 2)
        A, _ = torch.max(alpha, 2)
        tmp = torch.sum((alpha - A.expand(alpha.size())).exp_(), 2).log()
        log_normalizer = tmp + A
        alpha = (alpha - log_normalizer.expand(alpha.size())).exp_()
        mu = torch.cat(map(lambda x: x.unsqueeze(2), mu), 2)
        sigma = torch.cat(map(lambda x: x.unsqueeze(2), sigma), 2)
        sigma = torch.clamp(sigma, min=1e-6)
        
        return alpha, mu, sigma
        
    def sample(self, parents, ns=1):
        """ Given a setting of the observed (parent) random variables, sample values of the latents.
        
            If ns > 1, returns a tensor whose first dimension is ns, each containing a sample
            of size [batch_size, D_latent] """
        assert parents.size(1) == self.D_in - self.D_out
        original_batch_size = parents.size(0)
        if ns > 1:
            parents = parents.repeat(ns,1)
        batch_size = parents.size(0)
            
        
        # sample noise variables
        FloatTensor = torch.cuda.FloatTensor if parents.is_cuda else torch.FloatTensor
        latent = Variable(torch.zeros(batch_size, self.D_out))
        randvals = Variable(torch.FloatTensor(batch_size, self.D_out))
        torch.randn(batch_size, self.D_out, out=randvals.data);
        gumbel = Variable(torch.rand(batch_size, self.D_out, self.K).log_().mul_(-1).log_().mul_(-1))
        if parents.is_cuda:
            latent = latent.cuda()
            randvals = randvals.cuda()
            gumbel = gumbel.cuda()

        for d in xrange(self.D_out):
            full_input = torch.cat((parents, latent), 1)
            alpha, mu, sigma = self(full_input)
            _, z = torch.max(alpha.log() + gumbel, 2)
            one_hot = torch.zeros(alpha.size())
            if parents.is_cuda: one_hot = one_hot.cuda()
            one_hot = one_hot.scatter_(2, z.data, 1).squeeze_().byte()
            latent = Variable(randvals.data * sigma.data[one_hot].view(z.size()) + mu.data[one_hot].view(z.size()))
        if ns > 1:
            latent = latent.resize(ns, original_batch_size, self.D_out)
        return latent
        
    def logpdf(self, parents, values):
        """ Return the conditional log probability `ln p(values|parents)` """
        full_input = torch.cat((parents, values), 1)
        alpha, mu, sigma = self(full_input)
        eps = 1e-6 # need to prevent hard zeros
        alpha = torch.clamp(alpha, eps, 1.0-eps)
        
        const = sigma.pow(2).mul_(2*np.pi).log().mul_(0.5)
        normpdfs = (values[:,:,None].expand(mu.size()) - mu).div(sigma).pow(2).div_(2).add_(const).mul_(-1)
        lw = normpdfs + alpha.log()
#         print "norm", normpdfs, normpdfs.sum()
#         print "alph", alpha.log(), alpha.log().sum()
        # need to do log-sum-exp along dimension 2
        A, _ = torch.max(lw, 2)
        weighted_normal = (torch.sum((lw - A.expand(lw.size())).exp(), 2).log() + A).squeeze(2)
        return torch.sum(weighted_normal, 1)
