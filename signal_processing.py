import numpy as np
import scipy.stats
import scipy.signal
import scipy.stats

# chi square normality test
def chi2normal(y, alpha=0.05):
    cdf_x = np.sort(y)
    cdf_data = np.linspace(0, 1, len(y), endpoint=False)
    cdf_model = scipy.stats.norm.cdf(cdf_x, loc=y.mean(), scale=y.std())    
    chi2 = np.sum(np.square(cdf_data - cdf_model)/cdf_model)
    cv = scipy.stats.chi2.ppf(1 - alpha, len(y))
    return chi2 < cv

# entropy of discrete signal
def entropy(x):
    _, n = np.unique(x, return_counts=True)
    p = n/np.sum(n)
    return -np.sum(p*np.log2(p))

# self information of samples
def self_info(y):
    idx, n, _ = binpacking(y)
    p = n/n.sum()
    return -np.log2(p[idx-1])

# min max undersampling
def min_max_undersamp(x, y, n):
    L = len(y)
    nPad = int(n*np.ceil(L/n) - L)
    m = int((L + nPad)/n)
    x_pad = np.concatenate((x, np.zeros(nPad)))
    y_pad = np.concatenate((y, np.zeros(nPad)))
    x_stack = np.reshape(x_pad, (-1, n))
    y_stack = np.reshape(y_pad, (-1, n))
    y_mins = np.min(y_stack, axis=1)
    y_maxs = np.max(y_stack, axis=1)
    y_min_locs = np.argmin(y_stack, axis=1)
    y_max_locs = np.argmax(y_stack, axis=1)
    x_out = np.zeros(2*m)
    y_out = np.zeros(2*m)
    for i in range(m):
        yil = y_min_locs[i]
        yal = y_max_locs[i]
        min_before_max = yil < yal
        x_out[2*i] = x_stack[i, yil] if min_before_max else x_stack[i, yal]
        x_out[2*i+1] = x_stack[i, yal] if min_before_max else x_stack[i, yil]
        y_out[2*i] = y_mins[i] if min_before_max else y_maxs[i]
        y_out[2*i+1] = y_maxs[i] if min_before_max else y_mins[i]
    return x_out, y_out

# envelop reduce
def envelop_reduce(x, y):
    top_idx = scipy.signal.argrelmax(y)[0]
    bot_idx = scipy.signal.argrelmin(y)[0]
    if len(bot_idx) < len(top_idx):
        top_idx = top_idx[0:-1]
    elif len(bot_idx) > len(top_idx):
        bot_idx = bot_idx[0:-1]
    top_val = y[top_idx]
    top_pos = x[top_idx]
    bot_val = y[bot_idx]
    bot_pos = x[bot_idx]
    avg_val = np.mean(np.vstack((top_val, bot_val)), axis=0)
    avg_pos = np.mean(np.vstack((top_pos, bot_pos)), axis=0)
    return avg_pos, avg_val

# peak prominence and extend
def prominence(x, sign=1):
    prom = np.zeros(len(x))
    extend = np.zeros(len(x))
    x = x*sign
    for i in range(len(x)):
        if i == 0:
            left = -np.inf
            right = len(x)-1
            for j in range(i, len(x)):
                if x[j] > x[i]:
                    right = j
                    break
            rightRef = np.min(x[i:right])
            leftRef = 0
        elif i == len(x) - 1:
            right = np.inf
            left = 0
            for j in range(i, 0, -1):
                if x[j] > x[i]:
                    left = j
                    break   
            leftRef = np.min(x[left:i])
            rightRef = 0
        else:
            left = 0
            for j in range(i, 0, -1):
                if x[j] > x[i]:
                    left = j
                    break         
            leftRef = np.min(x[left:i])
            right = len(x)-1
            for j in range(i, len(x)):
                if x[j] > x[i]:
                    right = j
                    break
            rightRef = np.min(x[i:right])
        extend[i] = np.min([right - i, i - left])
        prom[i] = sign*(x[i] - np.max([rightRef, leftRef]))
    return prom, extend

# discretize data on the given range with the given number of bins
def discretize(x, d):
    reading = np.round(x*d)
    return reading/d

# find index of knee in 2d data
def find_knee(x, y):
    numerator = np.abs((x[-1] - x[0])*(y[0] - y) - (x[0] - x)*(y[-1] - y[0]))
    denominator = np.sqrt(np.square(x[-1] - x[0]) + np.square(y[-1] - y[0]))
    d = numerator/denominator
    return np.argmax(d)

# decimation for high downsampling factors
def decimate(x, q):
    max_q = 13
    if q <= max_q:
        return scipy.signal.decimate(x, q, zero_phase=True)
    else:
        y = x.copy()
        q_back = q
        while q_back > 1:
            for f in np.arange(max_q, 1, -1):
                if np.mod(q_back, f) == 0:
                    factor = f
            y = scipy.signal.decimate(y, int(factor), zero_phase=True)
            q_back = int(q_back/factor)
        return y
    
# make data continous
def make_continous(xp, yp):
    def fun(x):
        return np.interp(x, xp, yp)
    return fun

# find a minimal binning for data
def binpacking(x):
    N = x.size
    n, bins = np.histogram(x, bins=N)
    while np.any(n==0):
        N -= np.sum(n==0)
        d = (x.max() - x.min())/N
        bins = np.arange(x.min()-d/2, x.max()+d, d)
        n, _ = np.histogram(x, bins=bins)
    idx = np.digitize(x, bins)
    cent = bins[0:-1] + np.diff(bins)/2
    return idx, n, cent

# reduce data by rejecting low contributions to total variance
def variance_reduction(y):
    acc_var = np.square(y - y.mean())/y.var()/y.size
    acc_var_idx = np.argsort(acc_var)
    acc_var.sort()
    cum_var = np.cumsum(acc_var)
    cutoff_idx = find_knee(np.arange(cum_var.size), cum_var)
    cutoff = cum_var[cutoff_idx]
    keep_idx = acc_var_idx[cum_var > cutoff]
    discard_idx = acc_var_idx[cum_var < cutoff]
    keep_idx.sort()
    discard_idx.sort()
    return keep_idx, discard_idx, cutoff

# modified zscore using median instead of mean
def modified_zscore(x):
    med = np.median(x)
    mad = np.median(np.abs(med - x))
    k = 1/scipy.stats.norm.ppf(3/4)
    std = k*mad
    return (x - med)/std

# probability of each sample from the valued distribution
def sample_probability(x):
    uni, idx, n = np.unique(x, return_inverse=True, return_counts=True)
    p = n/np.trapz(n, uni)
    return p[idx]

# mean extrem value of gaussian distributed random variable depending on sample size
def mean_extreme_value(N):
    return np.sqrt(2*np.log(N))

# find peak by finding maximum between upper and lower bound
def find_peak(x, upper=1, lower=-1):
    trans = np.where(np.logical_or(x < lower, x > upper))[0]
    pre = trans[:-1]
    post = trans[1:]
    pos = np.where(np.logical_and(x[pre] < lower, x[post] > upper))[0]
    neg = np.where(np.logical_and(x[pre] > upper, x[post] < lower))[0]
    if len(pos) > 0:
        if pos[0] > neg[0]:
            neg = neg[1:]
        if len(pos) > len(neg):
            pos = pos[:-1]
        idx = np.zeros(len(pos), dtype=np.int32)
        for i in range(len(pos)):
            idx[i] = np.argmax(x[pre[pos[i]]:pre[neg[i]]+1]) + pre[pos[i]]
    else:
        idx = np.zeros(0, dtype=np.int32)
    return idx

# simulate n points of a pink noise of duration T
def simulate_pink_noise(n, T):
    n = int(n)
    f = np.fft.rfftfreq(n, d=T/n)
    C = 1.0/f
    C[0] = 0
    C = np.sqrt(C)
    phi = 2.0*np.pi*np.random.rand(len(f))
    C = C*np.exp(1j*phi)
    return np.fft.irfft(C)


from itertools import islice
from collections import deque
from bisect import bisect_left, insort

def running_median_insort(seq, window_size):
    seq = iter(seq)
    d = deque()
    s = []
    result = []
    for item in islice(seq, window_size):
        d.append(item)
        insort(s, item)
        result.append(s[len(d)//2])
    m = window_size // 2
    for item in seq:
        old = d.popleft()
        d.append(item)
        del s[bisect_left(s, old)]
        insort(s, item)
        result.append(s[m])
    return result
