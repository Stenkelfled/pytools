# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 14:16:41 2014

@author: stenkelfled
"""

import glob
from math import isnan
import numpy as np
import scipy.io
import scipy.signal

class Measurement:
    def __init__(self, data, time, length, Tinterval, Tstart=0):
        if type(data) != list:
            data = [data,]
        self.data = data
        self.time = time
        self.Length = length
        self.Tinterval = Tinterval
        self.Tstart = Tstart
        self.index = 0
        self.envelopes = dict()
#        if(source == "matlab"):
#            self.A = data["A"][:,0]
#            self.B = data["B"][:,0]
#            self.Length = data["Length"][0]
#            self.Tinterval = data["Tinterval"][0]
#            self.Tstart = data["Tstart"][0]
#            self.time = np.arange(self.Tstart, self.Tinterval*self.Length+self.Tstart, self.Tinterval, dtype=float)
#        elif(source == "ltspice"):
#            self.time = data[0]
#            self.A = data[1]
#            self.Length = len(self.time)
        
    def __getitem__(self, key):
        return self.data[key]
        
    def __setitem__(self, key, value):
        self.data[key] = value
        
    def __iter__(self):
        return self.data.__iter__()
        
    def next(self):
        return self.data.next()

    def min(self, key):
        return min(self[key])
        
    def max(self, key):
        return max(self[key])

    def calcAbs(self):
        for idx in xrange(0, len(self.data)):
            self[idx] = np.abs(self[idx])
            
    def normalize(self):
        for idx in xrange(0, len(self.data)):
            self[idx] /= self.max(idx)
            
    def calcEnvelope(self, key):
        self.envelopes[key] = abs(scipy.signal.hilbert(self[key],axis=0))
        
#            
#    def crop(self, start, end):
#        """ remove begin and/or end of the signal
#            @param: start: start-index of the signal part, that is meant to be kept
#            @param: end: end-index of the signal part, that is meant to be kept
#        """
#        self.A = self.A[start:end]
#        self.B = self.B[start:end]
#        self.time = self.time[start:end]
#        self.Tstart = self.time[0]
#        self.Length = len(self.A)
#        
#    def cropRatio(self, start=0, end=1):
#        """ remove begin and/or end of the signal
#            @param: start: start-ratio(0...1) of the signal part, that is meant to be kept
#            @param: end: end-ratio(1...0) of the signal part, that is meant to be kept
#            start=0 and end=1 -> do not crop anything
#        """
#        self.crop(int(start*self.Length), int(end*self.Length))
#        
    def shift(self, time):
        """shift the signal to right/left
            @param: time: time to shift in s. Negative Values: shift to left
        """
        self.time += time
#        
#    def getEnvelopeA(self):
#        return self.envelopeA
#        
#    def getPlotDataEnvelopeA(self, **kwargs):
#        return (self.getPlotData(self.envelopeA, **kwargs),)
#        
#    def getPlotDataEnvelopeAMaximum(self, value=0):
#        return (([self.maxtime],[value]),)
#            
    def getPlotData(self, key, start=0, end=1):
        data_start = int(round(self.Length*start))
        data_end = int(round(self.Length*end))
        return ((self.time[data_start:data_end], self.data[key][data_start:data_end]),)
        
    def getPlotDataSelection(self, keys, **kwargs):
        plot_data = []
        for key in keys:
            plot_data.extend(self.getPlotData(key, **kwargs))
        return plot_data
        
    def getPlotDataAll(self, **kwargs):
        return self.getPlotDataSelection(range(0, len(self.data)), **kwargs)
        
    def filterSignal(self, key, frequencies):
        if (isnan(self.Tinterval)):
            raise ValueError("There is no time interval. Maybe you have fed a LTSpice-file. In this case filtering is not supported")
        signal = self[key]
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
        
#        
#    def resetTime(self):
#        self.Tstart = 0
#        self.time = np.arange(self.Tstart, self.Tinterval*self.Length+self.Tstart, self.Tinterval, dtype=float)
		
def readMatfiles(path, isfile=False):
    measures = []
    if(isfile):
        files = [path]
    else:
        files = glob.glob(path+"\*.mat")
    for file in files:
        data = scipy.io.loadmat(file)
        Length = data["Length"][0]
        Tinterval = data["Tinterval"][0]
        Tstart = data["Tstart"][0]
        time = np.arange(Tstart, Tinterval*Length+Tstart, Tinterval, dtype=float)        
        meas = Measurement([data["A"][:,0], data["B"][:,0]], time, Length, Tinterval, Tstart)
        
        measures.append(meas)
    return measures
    
def readLTSpiceFile(path):
    ltfile = open(path, 'r')
    line = ltfile.readline().split('\t')
    data_type = line[0]
    time = []
    data = []
    if(data_type == 'time'):
        for foo in xrange(1, len(line)):
            data.append([])
        print data
        for line in ltfile:
            line = line.split('\t')
            time.append(np.float(line[0]))
            for idx in xrange(1, len(line)):
                data[idx-1].append(np.float(line[idx]))
    elif(data_type == 'Freq.'):
        for foo in xrange(1, len(line)):
            data.append([])
            data.append([])
        for line in ltfile:
            line = line.split('\t')
            time.append(np.float(line[0]))
            for idx in xrange(1, len(line)):
                line_sp = line[idx].split(',')
                ampli = np.float(line_sp[0].strip('dB()\n°'))
                phase = np.float(line_sp[1].strip('dB()\n°'))
                data[(idx-1)*2].append(ampli)
                data[(idx-1)*2+1].append(phase)
    else:
        ltfile.close()
        raise TypeError("lt-spice-type"+data_type+"is not supported. Please add.")
    ltfile.close()
    for idx in xrange(0, len(data)):
        data[idx] = np.array(data[idx])
    return Measurement(data, time, len(time), nan)
    
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
    time = np.array(graphs[0]["time"])
    data = []
    for graph in graphs:
        data.append(np.array(graph["data"]))
    return Measurement(data, time, graphs[0]["Record Length"], graphs[0]["Sample Interval"])
    
#==============================================================================
# for testing
#==============================================================================
if(__name__ == "__main__"):
    import plotNicely as pN
    pN.plt.close('all')
    #meas = readLTSpiceFile(r"F:\Studienarbeit\Simulation\US-Sender\sender_rechteck.txt")
    #meas = readLTSpiceFile(r"F:\Studienarbeit\Messungen\Messfilter\filter_bode.txt")
    #meas = readMatfiles(r'F:\Studienarbeit\Messungen\Schall\Gegenstand\20140630-0001_Inbus_08mm.mat', isfile=True)[0]
    meas = readTekFile(r"F:\Diplomarbeit\Graphs\grosses_Netzteil\Motor-fest_Strom_PWM50.csv")
    foo = pN.plot(meas.getPlotData(0))
    #foo = pN.plot(meas.getPlotDataAll())
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    