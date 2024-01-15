import numpy as np
import matplotlib.pyplot as plt



def autocorrelation(x):
    """Does the auto correlation of a signal X with no scaling"""
    acorr = np.zeros(len(x))
    x_per = np.concatenate([x,x])
    for k in range(len(x)):
        for n in range(len(x)):
            # Out of range elements are considered to be zero
            #if (n+k < len(x)):
            acorr[k] = x[n]*x_per[n+k] + acorr[k]

    return acorr

# Create a signal
start = 0
step = 0.001
Freq = 10
Freq2 = 20
N = 1000
t = np.arange(0,N)*step+start
x = 20*np.sin(2*np.pi*Freq*t) + 10*np.sin(2*np.pi*Freq2*t)+np.random.normal(0,2,N)

# Get autocorrelation
y = autocorrelation(x)

plt.figure(0)
plt.plot(x)
plt.plot(y)
# Computing PSD
y_psd = np.fft.fft(y)
# Computing ESD
y_esd = np.abs(np.fft.fft(x))**2
#y_log = 10*np.log10(y_psd)
plt.figure(1)
plt.plot(y_esd, label='Y')
#plt.figure(2)
plt.plot(y_psd, label='Auto')
plt.legend(loc="upper left")

#plt.psd(x, Fs=1/step, NFFT=len(x))
plt.show()