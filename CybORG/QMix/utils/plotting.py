import matplotlib.pyplot as plt
import os
import numpy as np

def plot_returns(returns, path):
    mean, sem = _prepare_returns(returns)
    bounds = (mean - 2 * sem, mean + 2 * sem)
    plt.plot(mean, color='black')
    plt.fill_between(range(len(mean)), *bounds, color='red', alpha=0.3)
    plt.xlabel('episode #')
    plt.ylabel('episode return (sum of agent discounted rewards)')
    plt.title('episode reward evolution')
    plt.savefig(os.path.join(path, 'returns.png'))
    plt.close()

def _prepare_returns(returns):
    mean = np.mean(returns, axis=0)
    sem = np.std(returns, axis=0) / np.sqrt(len(returns))
    return mean, sem