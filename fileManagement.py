# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 14:16:41 2014

@author: stenkelfled
"""

import glob
import numpy as np
import scipy.io
import scipy.signal

class Measurement:
    def __init__(self, data, source):
        if(source == "matlab"):
            self.A = data["A"][:,0]
            self.B = data["B"][:,0]
            self.Length = data["Length"][0]
            self.Tinterval = data["Tinterval"][0]
            self.Tstart = data["Tstart"][0]
            self.time = np.arange(self.Tstart, self.Tinterval*self.Length+self.Tstart, self.Tinterval, dtype=float)
        elif(source == "ltspice"):
            self.time = data[0]
            self.A = data[1]
            self.Length = len(self.time)

    def Amin(self):
        return min(self.A)
        
    def Amax(self):
        return max(self.A)
        
    def Bmin(self):
        return min(self.B)
        
    def Bmax(self):
        return max(self.B)

    def calcAbs(self):
        self.A = np.abs(self.A)
        self.B = np.abs(self.B)

    def calcEnvelope(self):
        self.envelopeA = abs(scipy.signal.hilbert(self.A,axis=0))
        
    def calcEnvelopeMaximum(self):
        if(hasattr(self,'envelopeA')):
            self.maxtime = self.time[np.argmax(self.envelopeA)]
            
    def crop(self, start, end):
        """ remove begin and/or end of the signal
            @param: start: start-index of the signal part, that is meant to be kept
            @param: end: end-index of the signal part, that is meant to be kept
        """
        self.A = self.A[start:end]
        self.B = self.B[start:end]
        self.time = self.time[start:end]
        self.Tstart = self.time[0]
        self.Length = len(self.A)
        
    def cropRatio(self, start=0, end=1):
        """ remove begin and/or end of the signal
            @param: start: start-ratio(0...1) of the signal part, that is meant to be kept
            @param: end: end-ratio(1...0) of the signal part, that is meant to be kept
            start=0 and end=1 -> do not crop anything
        """
        self.crop(int(start*self.Length), int(end*self.Length))
        
    def shift(self, time):
        """shift the signal to right/left
            @param: time: time to shift in s. Negative Values: shift to left
        """
        self.time += time
        
    def getEnvelopeA(self):
        return self.envelopeA
        
    def getPlotDataA(self, **kwargs):
        return (self.getPlotData(self.A, **kwargs),)
        
    def getPlotDataEnvelopeA(self, **kwargs):
        return (self.getPlotData(self.envelopeA, **kwargs),)
        
    def getPlotDataEnvelopeAMaximum(self, value=0):
        return (([self.maxtime],[value]),)
    
    def getPlotDataB(self, **kwargs):
        return (self.getPlotData(self.B, **kwargs),)
        
    def getPlotData(self, data, start=0, end=1):
        data_start = round(self.Length*start)
        data_end = round(self.Length*end)
        return (self.time[data_start:data_end], data[data_start:data_end])
        
    def getPlotDataAll(self, **kwargs):
        return (self.getPlotData(self.A, **kwargs),self.getPlotData(self.B, **kwargs))
        
    def filterA(self, frequencies):
        self.A = self.filterSignal(self.A, frequencies)
        
    def filterB(self, frequencies):
        self.B = self.filterSignal(self.B, frequencies)
    
    def filterSignal(self, signal, frequencies):
        for freq in frequencies:
            if(freq < 0):
                my_btype = "lowpass"
            else:
                my_btype = "highpass"
            (b,a) = scipy.signal.butter(4, np.array(2*abs(freq)/(1/self.Tinterval)), btype = my_btype)
            signal = scipy.signal.lfilter(b,a,signal,0)
        return signal
        
    def filterGetStandardFrequencies(self):
        FILTER_US_TP_FREQ = 60e3 #Bandpass for the input signal
        FILTER_US_HP_FREQ = 20e3 #Bandpass for the input signal
        return (-FILTER_US_TP_FREQ,FILTER_US_HP_FREQ)
        
    def normalizeA(self):
        self.A /= self.Amax()
        
    def resetTime(self):
        self.Tstart = 0
        self.time = np.arange(self.Tstart, self.Tinterval*self.Length+self.Tstart, self.Tinterval, dtype=float)
		
def readMatfiles(path, isfile=False):
    measures = []
    if(isfile):
        files = [path]
    else:
        files = glob.glob(path+"\*.mat")
    for file in files:
        measures.append(Measurement(scipy.io.loadmat(file), "matlab"))
    return measures
    
def readLTSpiceFile(path):
    data = list()
    ltfile = open(path, 'r')
    line = ltfile.readline()
    data_type = line.split('\t')[0]
    if(data_type == 'time'):
        for line in ltfile:
            data.append([np.float(x) for x in line.split('\t')])
    elif(data_type == 'Freq.'):
        for line in ltfile:
            line_sp = line.split('\t')
            freq = np.float(line_sp[0])
            line_sp = line_sp[1].split(',')
            ampli = np.float(line_sp[0].strip('dB()\n°'))
            phase = np.float(line_sp[1].strip('dB()\n°'))
            data.append([freq, ampli, phase])            
    else:
        ltfile.close()
        raise TypeError("lt-spice-type"+data_type+"is not supported. Please add.")
    ltfile.close()
    data = np.array(data)
    measures = []
    for i in xrange(1,data.shape[1]):
        measures.append(Measurement([data[1:,0], data[1:,i]], "ltspice"))
        
    return measures
    
def readTekFile(path):
    ELEM_PER_GRAPH = 6
    tekfile = open(path, 'r')
    line = tekfile.readline().split(',')
    graph_count = len(line)/ELEM_PER_GRAPH
    graphs = list()
    for i in xrange(graph_count):
        graphs.append({"time":list(),
                       "data":list()
        })
    tekfile.seek(0,0) #start from beginning!
    for line in tekfile:
        line = line.split(',')
        for i in xrange(graph_count):
            graph_line = line[i*ELEM_PER_GRAPH:(i+1)*ELEM_PER_GRAPH]
            if graph_line[0] != '':
                try:
                    value = float(graph_line[1])
                except ValueError:
                    pass
                else:
                    graphs[i][graph_line[0]] = value
            graphs[i]["time"].append(float(graph_line[3]))
            graphs[i]["data"].append(float(graph_line[4]))
    tekfile.close()
    return graphs
    
#==============================================================================
# for testing
#==============================================================================
if(__name__ == "__main__"):
    import plotNicely as pN
    pN.plt.close('all')
    #measures = readLTSpiceFile(r"F:\Studienarbeit\Simulation\US-Sender\sender_rechteck.txt")
    #measures = readLTSpiceFile(r"F:\Studienarbeit\Messungen\Messfilter\filter_bode.txt")
    meas = readMatfiles(r'F:\Studienarbeit\Messungen\Schall\Gegenstand\20140630-0001_Inbus_08mm.mat', isfile=True)[0]
    meas.shift(-1e-3)
    pN.plot(meas.getPlotDataA())
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    