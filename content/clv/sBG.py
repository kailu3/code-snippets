import numpy as np
import pandas as pd
from scipy.optimize import minimize


def compute_probabilities(alpha, beta, num_periods):
    '''Compute the probability of a random person churning for each time period'''
    p = [alpha / (alpha + beta)]
    for t in range(2, num_periods + 1):
        p.append((beta + t - 2) / (alpha + beta + t - 1) * p[t - 2])
    return p


def pct_dying(data, num_periods):
    '''Compute the number of people who churn for each time period'''
    n = [1 - data[0]]
    for t in range(1, num_periods):
        n.append(data[t - 1] - data[t])
    return n


def log_likelihood(alpha, beta, data):
    '''Objective function that we need to maximize to get best alpha and beta parameters
    **Computed log-likelihood will be 1/n smaller than the actual log-likelihood (n = original sample size)**
    '''
    if alpha <= 0 or beta <= 0:
        return -99999
    probabilities = np.array(compute_probabilities(alpha, beta, len(data)))
    percent_dying = np.array(pct_dying(data, len(data)))
    
    return np.sum(np.log(probabilities) * percent_dying) + data[-1] * np.log(1 - np.sum(probabilities))


def maximize(data):
    '''Maximize log-likelihood by searching for best (alpha, beta) combination'''
    func = lambda x: -log_likelihood(x[0], x[1], data) # x is a tuple (alpha, beta)
    x0 = np.array([100., 100.])
    res = minimize(func, x0, method='Nelder-Mead', options={'xtol': 1e-8, 'disp': False})

    return res.x


def forecast(data, num_periods):
    '''Forecast num_periods from the data using the sBG model'''
    alpha, beta = maximize(data)
    probabilities = compute_probabilities(alpha, beta, num_periods)
    
    expected_alive = [1 - probabilities[0]]
    for t in range(1, num_periods):
        expected_alive.append(1 - np.sum(probabilities[0:t+1]))
    
    return pd.Series(expected_alive)


def forecast_dataframe(data, num_periods):
    '''Creates dataframe with forecast with additional performance columns'''
    forecast_column = forecast(data, num_periods)
    actual_column = pd.Series(data)
    
    period = pd.DataFrame({'Period': np.arange(1, np.max([len(data) + 1, num_periods + 1]))})
    actual = pd.DataFrame({'Actual': actual_column})
    the_forecast = pd.DataFrame({'Forecast': forecast_column})
    
    df = pd.concat([period, actual, the_forecast], axis=1)
    
    # Compute pct error as well
    df['pct_error'] = np.abs(df['Actual'] - df['Forecast']) / df['Actual'] * 100   
    
    return df