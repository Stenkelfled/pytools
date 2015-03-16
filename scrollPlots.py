# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 10:16:06 2014

@author: stenkelfled
"""
import fileManagement as fM
import plotNicely as pN

pN.plt.close('all')

path = r"F:\Studienarbeit\Messungen\Schall\Entfernung_Winkel\07_ohne Kegel\290_100_1390"
measures = fM.readMatfiles(path)

for meas in measures:
    meas.filterA(meas.filterGetStandardFrequencies())#(-FILTER_US_TP_FREQ,))
    meas.calcEnvelope()
    meas.calcEnvelopeMaximum()

a = pN.plot([(meas.getPlotDataA()[0], meas.getPlotDataEnvelopeA()[0], meas.getPlotDataEnvelopeAMaximum()[0]) for meas in measures], multiple_traces=True, scroll=True)











































