import torch
import hw2_utils as utils
import matplotlib.pyplot as plt
import sklearn.datasets as datasets
from sklearn.model_selection import train_test_split

''' Translation of mjt-old's solutions to PyTorch '''
def gaussian_theta(X, y):
    '''
    Arguments:
        X (S x N FloatTensor): features of each object
        y (S LongTensor): label of each object, y[i] = 0/1

    Returns:
        mu (2 x N Float Tensor): MAP estimation of mu in N(mu, sigma2)
        sigma2 (2 x N Float Tensor): MAP estimation of mu in N(mu, sigma2)

    '''

    X0 = X[y == 0]
    X1 = X[y == 1]


    mu0 = torch.mean(X0, dim=0)
    mu1 = torch.mean(X1, dim=0)

    sigma20 = torch.var(X0, dim=0, unbiased=False)
    sigma21 = torch.var(X1, dim=0, unbiased=False)

    mu = torch.stack((mu0, mu1))
    sigma2 = torch.stack((sigma20, sigma21))

    return mu, sigma2


def gaussian_p(y):
    '''
    Arguments:
        y (S LongTensor): label of each object

    Returns:
        p (float or scalar Float Tensor): MLE of P(Y=0)

    '''
    p = torch.mean((y == 0).float())
    return p


def gaussian_classify(mu, sigma2, p, X):
    '''
    Arguments:
        mu (2 x N Float Tensor): returned value #1 of `gaussian_MAP`
        sigma2 (2 x N Float Tensor): returned value #2 of `gaussian_MAP`
        p (float or scalar Float Tensor): returned value of `bayes_MLE`
        X (S x N LongTensor): features of each object for classification, X[i][j] = 0/1

    Returns:
        y (S LongTensor): label of each object for classification, y[i] = 0/1

    '''
    X_expanded = X.unsqueeze(1)
    mu_expanded = mu.unsqueeze(0)
    sigma2_expanded = sigma2.unsqueeze(0)

    log_gaussian = -0.5 * torch.log(2 * torch.pi * sigma2_expanded) \
                   - ((X_expanded - mu_expanded) ** 2) / (2 * sigma2_expanded)


    log_likelihood = torch.sum(log_gaussian, dim=2)

    p = torch.as_tensor(p, dtype=mu.dtype, device=mu.device)
    log_prior = torch.stack((torch.log(p), torch.log(1 - p)))
    log_posterior = log_likelihood + log_prior

    y_pred = torch.argmax(log_posterior, dim=1).long()

    return y_pred
