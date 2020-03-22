import numpy as np
import pandas as pd
from scipy.optimize import minimize
import scipy.special as sc

def shape(df):
    '''TODO: Shapes transactional data into recency and frequency'''
    pass

def i_likelihood(alpha, beta, gamma, delta, x, t_x, n):
    '''Calculates the individual likelihood for a person
    given paramters, frequency, recency, and periods
    '''
    # Summation component
    summation = [sc.beta(alpha+x, beta+t_x-x+i) / sc.beta(alpha, beta)
                 * sc.beta(gamma+1, delta+t_x+i) / sc.beta(gamma, delta) 
                 for i in range(0, n-t_x)]
    
    # First component
    return (sc.beta(alpha+x, beta+n-x) / sc.beta(alpha, beta)
            * sc.beta(gamma, delta+n) / sc.beta(gamma, delta)
            + sum(summation))

def log_likelihood(df, alpha, beta, gamma, delta):
    '''Computes total log-likelihood for given parameters'''
    # Get frequency, recency, and periods lists
    frequency = df.frequency.to_list()
    recency = df.recency.to_list()
    periods = df.periods.to_list()
    
    # Compute individual likelihood first
    indiv_llist = [i_likelihood(alpha, beta, gamma, delta, frequency[i], recency[i], periods[i]) 
                   for i in range(0, len(frequency))]
    
    # Multiply with num_obs
    num_obs = np.array(df.num_obs.to_list())
    
    return np.sum(num_obs * np.log(np.array(indiv_llist)))


def maximize(df):
    '''Maximize log-likelihood by searching for best 
    (alpha, beta, gamma, delta) combination'''
    func = lambda x: -log_likelihood(df, x[0], x[1], x[2], x[3])
    x0 = np.array([1., 1., 1., 1.])
    res = minimize(func, x0, method='Nelder-Mead', options={'xtol': 1e-8, 'disp': False})
    return res.x


def prob_alive(df, x, t_x, p):
    '''Probability for a customer with transaction
    history (x, t_x, n) to be alive at time period p'''
    alpha, beta, gamma, delta = maximize(df)
    n = df.periods.iloc[0]
    indiv_ll = i_likelihood(alpha, beta, gamma, delta, x, t_x, n)
    return (sc.beta(alpha+x, beta+n-x) / sc.beta(alpha, beta)
            * sc.beta(gamma, delta+p) / sc.beta(gamma, delta)
            / indiv_ll)

def prob_alive_df(df, p):
    '''List of probabilities for a customer to be alive
    for all the combinations of (x, t_x)'''
    alpha, beta, gamma, delta = maximize(df)
    n = df.periods.iloc[0]
    
    x_list = df.frequency.to_list()
    t_x_list = df.recency.to_list()
    
    p_list = [sc.beta(alpha+x_list[i], beta+n-x_list[i]) 
               / sc.beta(alpha, beta)
               * sc.beta(gamma, delta+p) / sc.beta(gamma, delta)
               / i_likelihood(alpha, beta, gamma, delta, x_list[i], t_x_list[i], n) 
               for i in range(0, len(x_list))]
    
    return pd.DataFrame({'frequency': x_list,
                         'recency': t_x_list,
                         'p_alive': p_list})

def expected_count(df, n):
    '''Calculates the mean number of transactions
    occurring across the first n transaction opportunities
    '''
    alpha, beta, gamma, delta = maximize(df)
    E_x = [alpha / (alpha+beta) 
           * delta / (gamma-1)
           * (1-sc.gamma(gamma+delta)/sc.gamma(gamma+delta+i)
           * sc.gamma(1+delta+i)/sc.gamma(1+delta))
           for i in range(1, n+1)]
    return pd.DataFrame({'n': np.arange(1, n+1),
                         'E[X(n)]': E_x,
                        })

def pmf(df, x, alpha, beta, gamma, delta):
    '''probabilility of having x transactions in the dataset
    (essentially the pmf)'''
    n = df.periods.iloc[0]
    
    # Summation component
    summation = [sc.comb(i, x)
                 * sc.beta(alpha+x, beta+t_x-x+i) / sc.beta(alpha, beta)
                 * sc.beta(gamma+1, delta+t_x+i) / sc.beta(gamma, delta) 
                 for i in range(0, n-t_x)]
    
    return (sc.comb(n, x)
            * sc.beta(alpha+x, beta+n-x) / sc.beta(alpha, beta)
            * sc.beta(gamma, delta+n) / sc.beta(gamma, delta)
            + sum(summation))


def pmf_df(df):
    '''
    Creates a dataframe for all possible x's
    '''
    alpha, beta, gamma, delta = maximize(df)
    p_x = [pmf(df, i, alpha, beta, gamma, delta)
           for i in range(0, df.periods.loc[0]+1)]
    
    return pd.DataFrame({'x': np.arange(0, df.periods.loc[0]+1),
                         'p': p_x})


def cond_expectations(alpha, beta, gamma, delta, x, t_x, n, p):
    '''The expected number of future transactions
    across the next p transaction opportunities by
    a customer with purchase history (x, t_x, n)'''    
    return (1/i_likelihood(alpha, beta, gamma, delta, x, t_x, n)
            * sc.beta(alpha+x+1, beta+n-x)/sc.beta(alpha, beta)
            * delta/(gamma-1)
            * sc.gamma(gamma+delta)/sc.gamma(1+delta)
            * (sc.gamma(1+delta+n)/sc.gamma(gamma+delta+n)
             - sc.gamma(1+delta+n+p)/sc.gamma(gamma+delta+n+p))
           )


def cond_expectations_df(df, p):
    '''Creates a dataframe for CE for each
    (x, t_x, n) combination'''
    alpha, beta, gamma, delta = maximize(df)
    x_list = df.frequency.to_list()
    t_x_list = df.recency.to_list()
    n_list = df.periods.to_list()
    
    ce_list = [cond_expectations(alpha, beta, gamma, delta,
                                 x_list[i], t_x_list[i], n_list[i], p)
               for i in range(0, len(x_list))]
    
    return pd.DataFrame({'frequency': x_list,
                         'recency': t_x_list,
                         'n': n_list,
                         'ce': ce_list})


def marg_posterior_p(alpha, beta, gamma, delta, x, t_x, n, p):
    '''The marginal posterior distribution of p'''
    b1 = (p**(alpha+x-1) * (1-p)**(beta+n-x-1)
         / sc.beta(alpha, beta)
         * sc.beta(gamma, delta+n)/sc.beta(gamma, delta))
    b2 = [(p**(alpha+x-1) * (1-p)**(beta+t_x-x+i-1)
          / sc.beta(alpha, beta)
          * sc.beta(gamma+1, delta+t_x+i)
          / sc.beta(gamma, delta))
          for i in range(0, n-t_x)]
    return (b1 + sum(b2)) / ilikelihood(alpha, beta, gamma, delta, x, t_x, n)


def marg_posterior_theta(alpha, beta, gamma, delta, x, t_x, n, theta):
    '''The marginal posterior distribution of theta'''
    c1 = (sc.beta(alpha+x, beta+n-x) / sc.beta(alpha, beta)
          * theta**(gamma-1) * (1-theta)**(delta+n-1)
          / sc.beta(gamma, delta))
    c2 = [(sc.beta(alpha+x, beta+t_x-x+i) / sc.beta(alpha, beta)
          * theta**gamma * (1-theta)**(delta+t_x+i-1) / sc.beta(gamma, delta))
          for i in range(0, n-t_x)]
    return (c1 + sum(c2)) / ilikelihood(alpha, beta, gamma, delta, x, t_x, n)