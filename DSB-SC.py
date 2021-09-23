# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 19:52:35 2019

@author: Kostas Ziovas
Signal Processing Project 2019
"""

import numpy as np
import matplotlib.pyplot as plt


"""(a) DSB-SC MODULATION"""
# Fixing random state for reproducibility
np.random.seed(19680801)

dt = 0.01
t = np.arange(0, 30, dt)
#nse1 = np.random.randn(len(t))                 # white noise 1
#nse2 = np.random.randn(len(t))                 # white noise 2

# Two signals and their DSB-SC MODULATION
s1 = np.cos(2 * np.pi * t)
s2 = np.cos(2 * np.pi * 4 * t) 
DSBSC=s1*s2

fig, axs = plt.subplots(2, 1)
axs[0].plot(t, s1, t, s2)
axs[0].set_xlim(0, 2)
axs[0].set_xlabel('time')
axs[0].set_ylabel('s1 and s2')
axs[0].grid(True)

axs[1].plot(t,DSBSC)
axs[1].set_xlim(0, 2)
axs[1].set_ylabel('DSBSC')

#cxy, f = axs[1].cohere(s1, s2, 256, 1. / dt)
#axs[1].set_ylabel('coherence')

fig.tight_layout()
plt.show()

#The fft of the same signals
fts1=np.fft.fft(s1)
fts2=np.fft.fft(s2)
ftDSB=np.fft.fft(DSBSC)
freq = np.fft.fftfreq(len(s1),dt)


fig, axs = plt.subplots(2, 1)
axs[0].plot(freq, fts1, freq, fts2)
axs[0].set_xlim(-8, 8)
axs[0].set_xlabel('frequency')
axs[0].set_ylabel('ft-s1 and ft-s2')
axs[0].grid(True)

axs[1].plot(freq,ftDSB)
axs[1].set_xlim(-8, 8)
axs[1].set_xlabel('frequency')
axs[1].set_ylabel('ft-DSBSC')


"""(b) DSB-LC m=1 MODULATION"""
#XXXXXXXXXXXXXXXXXXXXXXXXXXX Define AM modulation Index XXXXXXXXXXXXXXXXXXX
#example: m=1 means 100% modulation
#m=input(' Enter the value of modulation index (m) = ');
m=1; # for 100% modulation


#XXXXXXXXXXXXXXXXX modulating signal generation XXXXXXXXXXXXXXXXXXXXXXXXXX
Am=1 # Amplitude of modulating signal
fa=1 # Frequency of modulating signal
#Ta=1/fa; % Time period of modulating signal
#t=0:Ta/999:6*Ta; # Total time for simulation
dt = 0.01
t = np.arange(0, 30, dt)


ym=Am*np.cos(2*np.pi*fa*t)# Equation of modulating signal

fig, axs = plt.subplots(2, 1)
fig.tight_layout()
#axs[0].plot(t, ym)
#axs[0].set_xlim(0, 4)
#axs[0].set_xlabel('time')
#axs[0].set_ylabel('s1')
#axs[0].grid(True)


#XXXXXXXXXXXXXXXXXXXXX carrier signal generation XXXXXXXXXXXXXXXXXXXXXXXXXX
Ac=Am/m# Amplitude of carrier signal [ where, modulation Index (m)=Am/Ac ]
fc=fa*4# Frequency of carrier signal
#Tc=1/fc;# Time period of carrier signal

yc=Ac*np.cos(2*np.pi*fc*t)#Equation of carrier signal

axs[0].plot(t,ym,t, yc)
axs[0].set_xlim(0, 4)
#axs[1].set_xlabel('time')
axs[0].set_ylabel('s1 and s2')
axs[0].grid(True)

#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX AM Modulation XXXXXXXXXXXXXXXXXXXXXXXXXXXXX 

yDSBLC=Ac*(1+m*np.cos(2*np.pi*fa*t))*np.cos(2*np.pi*fc*t) # Equation of DSBLC

axs[1].plot(t, yDSBLC)
axs[1].set_xlim(0, 4)
axs[1].set_xlabel('time')
axs[1].set_ylabel('DSBLC')
axs[1].grid(True)


#The fft of the same signals
fts1=np.fft.fft(ym)
fts2=np.fft.fft(yc)
ftDSB=np.fft.fft(yDSBLC)
freq = np.fft.fftfreq(len(s1),dt)


fig, axs = plt.subplots(2, 1)
axs[0].plot(freq, fts1, freq, fts2)
axs[0].set_xlim(-8, 8)
axs[0].set_xlabel('frequency')
axs[0].set_ylabel('ft-s1 and ft-s2')
axs[0].grid(True)

axs[1].plot(freq,ftDSB)
axs[1].set_xlim(-8, 8)
axs[1].set_xlabel('frequency')
axs[1].set_ylabel('ft-DSBLC')


"""(c) SSB-SC+ MODULATION"""

# Message Signal
dt = 0.01
t = np.arange(0, 30, dt)
Am=1 # Amplitude of modulating signal
fa=1 # Frequency of modulating signal
ym=Am*np.cos(2*np.pi*fa*t)# Equation of modulating signal 

Ac=1# Amplitude of carrier signal 
fc=4# Frequency of carrier signal
yc=Ac*np.cos(2*np.pi*fc*t)#Equation of carrier signal

ySSBSC=np.cos(2*np.pi*(fc+fa)*t)*(Am*Ac)/2 # Equation of SSBSC

fig, axs = plt.subplots(2, 1)
fig.tight_layout()

axs[0].plot(t,ym,t, yc)
axs[0].set_xlim(0, 4)
#axs[1].set_xlabel('time')
axs[0].set_ylabel('s1 and s2')
axs[0].grid(True)

axs[1].plot(t, ySSBSC)
axs[1].set_xlim(0, 4)
axs[1].set_xlabel('time')
axs[1].set_ylabel('SSBSC+')
axs[1].grid(True)


#The fft of the same signals
fts1=np.fft.fft(ym)
fts2=np.fft.fft(yc)
ftDSB=np.fft.fft(ySSBSC)
freq = np.fft.fftfreq(len(s1),dt)


fig, axs = plt.subplots(2, 1)
axs[0].plot(freq, fts1, freq, fts2)
axs[0].set_xlim(-8, 8)
axs[0].set_xlabel('frequency')
axs[0].set_ylabel('ft-s1 and ft-s2')
axs[0].grid(True)

axs[1].plot(freq,ftDSB)
axs[1].set_xlim(-8, 8)
axs[1].set_xlabel('frequency')
axs[1].set_ylabel('ft-SSBSC+')

"""(c) SSB-SC- MODULATION"""

# Message Signal
dt = 0.01
t = np.arange(0, 30, dt)
Am=1 # Amplitude of modulating signal
fa=1 # Frequency of modulating signal
ym=Am*np.cos(2*np.pi*fa*t)# Equation of modulating signal 

Ac=1# Amplitude of carrier signal 
fc=4# Frequency of carrier signal
yc=Ac*np.cos(2*np.pi*fc*t)#Equation of carrier signal

ySSBSC=np.cos(2*np.pi*(fc-fa)*t)*(Am*Ac)/2 # Equation of SSBSC

fig, axs = plt.subplots(2, 1)
fig.tight_layout()

axs[0].plot(t,ym,t, yc)
axs[0].set_xlim(0, 4)
#axs[1].set_xlabel('time')
axs[0].set_ylabel('s1 and s2')
axs[0].grid(True)

axs[1].plot(t, ySSBSC)
axs[1].set_xlim(0, 4)
axs[1].set_xlabel('time')
axs[1].set_ylabel('SSBSC-')
axs[1].grid(True)


#The fft of the same signals
fts1=np.fft.fft(ym)
fts2=np.fft.fft(yc)
ftDSB=np.fft.fft(ySSBSC)
freq = np.fft.fftfreq(len(s1),dt)


fig, axs = plt.subplots(2, 1)
axs[0].plot(freq, fts1, freq, fts2)
axs[0].set_xlim(-8, 8)
axs[0].set_xlabel('frequency')
axs[0].set_ylabel('ft-s1 and ft-s2')
axs[0].grid(True)

axs[1].plot(freq,ftDSB)
axs[1].set_xlim(-8, 8)
axs[1].set_xlabel('frequency')
axs[1].set_ylabel('ft-SSBSC-')