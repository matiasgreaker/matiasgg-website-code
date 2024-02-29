import numpy as np
import matplotlib.pyplot as plt



def autocorrelation(x, mode="periodic"):
    """ Does the auto correlation of a signal X with no scaling
        mode -- If 'periodic' x is assumed to repeat every N = length(x)
                If 'zero' x is assumed zero out of its bound
    """
    N = len(x)
    # Check if complex or not
    if np.iscomplexobj(x):
        comp = True
        acorr = np.zeros(N)+1j*np.zeros(N)
    else:
        comp = False
        acorr = np.zeros(N)
    # Check mode
    if (mode=="periodic"):
        x_per = np.concatenate([x,x])
    else:
        x_per = np.concatenate([x,np.zeros(N)])
    # Do autocorrelation
    for k in range(N):
        for n in range(N):
            if comp:
                acorr[k] = np.conjugate(x[n])*x_per[n+k] + acorr[k]
            else:
                acorr[k] = x[n]*x_per[n+k] + acorr[k]
    return acorr



def show_weiner_khinchin():
    # Create an arbitrary signal
    start = 0
    step = 0.001
    Freq = 14
    Freq2 = 50
    N = 1000
    t = np.arange(0,N)*step+start
    # Create a complex signal
    x = 20*np.exp(1j*2*np.pi*Freq*t) + 10*np.exp(1j*2*np.pi*Freq2*t)

    ## Proof
    # Calculate DFT of autocorrelation
    r = autocorrelation(x)
    r_fft = np.fft.fft(r)

    # Calculate Power of the DFT
    x_fft = np.fft.fft(x);
    x_psd = np.abs(x_fft)**2

    # Plot
    plt.figure(1)
    plt.title("|DFT(x)|^2 = DFT(r)")
    plt.plot(x_psd, label='|DFT(x)|^2')
    plt.plot(r_fft, label='DFT(r)')
    plt.legend(loc="upper left")

    plt.show()



#show_weiner_khinchin()
# 
start = 0
step = 0.001
Freq = 14
Freq2 = 50
N = 1000000
t = np.arange(0,N)*step+start
noise = np.random.normal(size=N)+1j*np.random.normal(size=N)
x = 20*np.exp(1j*2*np.pi*Freq*t) + 10*np.exp(1j*2*np.pi*Freq2*t)


sum = np.zeros(N+1)+1j*np.zeros(N+1)
for i in range(N):
    sum[i+1] = np.real(np.conjugate(noise[i]))*np.real(x[i]) + sum[i]


plt.plot(np.abs(sum[1:-1])/N)
print(sum[-1])
plt.show()

""" # Get autocorrelation
y = autocorrelation(x)
# Computing PSD
y_psd = np.fft.fft(y)
# Computing ESD
y_esd = np.abs(np.fft.fft(x))**2

print("Ratio if periodic: " + str(sum(y_psd/y_esd)/N))

# Get autocorrelation
y = autocorrelation(x, mode='zero')
# Computing PSD
y_psd = np.fft.fft(y)
# Computing ESD
y_esd = np.abs(np.fft.fft(x))**2
print("Ratio if zero: " + str(sum(y_psd/y_esd)/N))


y = autocorrelation(x)
plt.figure(0)
plt.plot(x)
plt.plot(y/N)
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
 """