import numpy as np
import matplotlib.pyplot as plt

def exponential_timeSeries(end_time, dt, SEED, std_dev, cor_len, max_freq):
    t = np.linspace(0,end_time+dt,int((end_time+dt) / dt))
    R_t = std_dev**2 * np.exp(-np.abs(t)/cor_len)
    
    N = len(t) # number of Fourier sample points
    h = t[1]-t[0] # grid spacing, nominally h=0.1 km for r=1
    L = N*h # profile length
    
    # white noise, unit normal distribution
    randomGen = np.random.default_rng(seed=SEED)
    y = randomGen.normal(size=N)
    # scale so PSD has unit amplitude
    y = y*np.sqrt(N/L)
    
    omega = 2 * np.pi * np.fft.fftfreq(N, h)
    Y = np.fft.fft(y)*h
    
    # apply fractal PSD
    PSD_expon = np.fft.fft(R_t) * h
    PSD_expon[np.abs(omega) > max_freq * 2 * np.pi] = 0
    
    Y_expon = Y * np.sqrt(PSD_expon)
    Y_expon[0] = 0
    Y_expon[-1] = 0
    PSDy_expon = np.abs(Y_expon)**2/L
    
    # transforming back into time domain
    y_expon = np.fft.ifft(Y_expon) / h
    y_expon = np.real(y_expon) - np.real(y_expon)[0]
    
    return y_expon, PSDy_expon, t

seeds = np.arange(10)
expon = []
expon_PSD = []

for SEED in seeds:
    exponential, psd_exponential, t = exponential_timeSeries(20*60, 0.04, SEED, 0.02, 10, 0.25)
    expon.append(exponential)
    expon_PSD.append(psd_exponential)
    np.savetxt('exponentialCorrelation_cVF_seed'+str(SEED)+'.txt', exponential)
np.savetxt('exponentialCorrelation_time.txt', t)

