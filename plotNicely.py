# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 14:16:41 2014

@author: stenkelfled
"""
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as pat

import numpy as np

from threading import Timer
    
#==============================================================================
class xy_data:
    
    def __init__(self,x,y):
        self.x = x
        self.y = y
        
#==============================================================================
class BoxZoom:
    
    def __init__(self):
        self.is_active = False        
        self.start_pos = None
        self.patch = None
        
    def start(self, event):
        self.is_active = True
        self.start_pos = xy_data(event.xdata, event.ydata)
        self.patch = None
        
    def moving(self, event):
        if( not self.is_active):
            return
        if(plt.gca() != event.inaxes):
            return
        if(self.patch != None):
            plt.gca().patches.remove(self.patch)
        verts = [(self.start_pos.x, self.start_pos.y),
                 (self.start_pos.x, event.ydata),
                 (event.xdata, event.ydata),
                 (event.xdata, self.start_pos.y),
                 (self.start_pos.x, self.start_pos.y)]
        codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
        path = Path(verts, codes)
        self.patch = pat.PathPatch(path, fill=False)
        plt.gca().add_patch(self.patch)
        plt.gcf().canvas.draw()
        
    def end(self, event):
        if(not self.is_active):
            return
        if(plt.gca() != event.inaxes):
            event.xdata = self.patch.get_path().vertices[2][0]
            event.ydata = self.patch.get_path().vertices[2][1]

        if( event.xdata > self.start_pos.x):
            plt.gca().set_xlim(self.start_pos.x, event.xdata)
        elif( event.xdata < self.start_pos.x):
            plt.gca().set_xlim(event.xdata, self.start_pos.x)
            
        if( event.ydata > self.start_pos.y):
            plt.gca().set_ylim(self.start_pos.y, event.ydata)
        elif( event.ydata < self.start_pos.y):
            plt.gca().set_ylim(event.ydata, self.start_pos.y)
        self.abort()
        return
        
    def abort(self):
        if(not self.is_active):
            return
        self.is_active = False
        if(self.patch != None):
            plt.gca().patches.remove(self.patch)
        plt.gcf().canvas.draw()

#==============================================================================
class Pan:
    
    def __init__(self):
        self.is_active = False
        self.pos = None
        self.lims = None
        self.start_lims = None
        
    def start(self, event):
        self.is_active = True
        self.pos = xy_data(event.xdata, event.ydata)
        self.lims = xy_data(plt.gca().get_xlim(), plt.gca().get_ylim())
        self.start_lims = xy_data(plt.gca().get_xlim(), plt.gca().get_ylim())
        
    def moving(self, event):
        if(not self.is_active):
            return
        if(plt.gca() != event.inaxes):
            return
        dist = xy_data(self.pos.x-event.xdata, self.pos.y-event.ydata)
        plt.gca().set_xlim( self.lims.x[0]+dist.x, self.lims.x[1]+dist.x )
        plt.gca().set_ylim( self.lims.y[0]+dist.y, self.lims.y[1]+dist.y )
        plt.gcf().canvas.draw()
        self.pos = xy_data(event.xdata+dist.x, event.ydata+dist.y)
        self.lims = xy_data(plt.gca().get_xlim(), plt.gca().get_ylim())
        
    
    def end(self, event):
        self.is_active = False

    def abort(self):
        if(not self.is_active):
            return
        self.is_active = False
        plt.gca().set_xlim(self.start_lims.x)
        plt.gca().set_ylim(self.start_lims.y)
        plt.gcf().canvas.draw()

#==============================================================================
# !!!IMPORTANT!!! call: a = plot(...); only plot(...) does not work
#==============================================================================
class plot:
        
    def __init__(self,data, axes = None, right_axes = None, hold = False, multiple_traces = False, line_style = ["b-","r-","g-","k-","c-","m-"], line_width = [1], markersize = 12, axis_style = "auto", size=None, xpowlim=(0,1), ypowlim=(0,1), scroll=False, additional_limits=None, xmul=1, ymul=1, xlim=None, ylim=None):
        """
            @param: data: ((x,y), (x,y))
            @param: axes: axes to plot in, or None
            @param: right_axes=[plot instance]: plot in same figure, but make new axis with ticks on right side
            @param: size: the size of the plot window in !!mm!!
            @param: multiple_traces: data: list of traces, one trace: ((x,y), (x,y)) !!only in scroll mode!!
        """
        ##copy input data to object
        matplotlib.rc('text', usetex = True)
        self.data = data
        self.line_style = line_style
        self.line_width = line_width
        self.axis_style = axis_style
        self.size = None        
        if(size!=None):
            self.size=size[0]/25.4, size[1]/25.4
        self.powlim = xy_data(xpowlim, ypowlim)
        self.scroll = scroll
        self.multiple_traces = multiple_traces
        self.markersize = markersize
        self.additional_limits = additional_limits
        self.xmul=xmul
        self.xlim=xlim
        self.ymul=ymul
        self.ylim=ylim
        
        ##generate plot output
        self.current_plot = 0
        if(right_axes == None):
            if(axes == None):
                if(hold == True):
                    self.figure = plt.gcf()
                    self.axes = plt.gca()
                else:
                    self.figure = plt.figure()
                    self.axes = self.figure.add_subplot(1,1,1)
            else:
                self.figure = plt.gcf()
                self.axes = axes
        else:
            self.figure = right_axes.figure
            self.axes = right_axes.axes.twinx()
            #self.axes.yticks
        if(self.scroll):
            self.plotCurrent()
            self.figure.canvas.mpl_connect('key_press_event', self.onKeyDownHandler)
            self.figure.canvas.mpl_connect('key_release_event', self.onKeyUpHandler)
        else:
            self.plotAll()

        #adjust ylabels            
        if(right_axes != None):
            pass
#            left_ticks = right_axes.axes.get_yticks()
#            print left_ticks
#            right_ticks = self.axes.get_yticks()
#            print right_ticks
#            new_right_ticks = np.linspace(right_ticks[0], right_ticks[-1], len(left_ticks))
#            print new_right_ticks
#            self.axes.set_yticks(new_right_ticks)
            #self.axes.yaxis.set_ticklabels(["%e" % val for val in new_right_ticks])
#            plt.plot()
        
        ##stuff for zooming
        self.box_zooming = BoxZoom()
        self.panning = Pan()
        
        ##connect handlers
        self.figure.canvas.mpl_connect('button_press_event', self.onMouseDownHandler)
        self.figure.canvas.mpl_connect('button_release_event', self.onMouseUpHandler)
        self.figure.canvas.mpl_connect('motion_notify_event', self.onMouseMoveHandler)
        self.figure.canvas.mpl_connect('scroll_event', self.onMouseScrollHandler)
        
        self.scroll_step = 0
        self.resetLineProperties()
        
    def apply_powlim(self):
        self.axes.xaxis.get_major_formatter().set_powerlimits(self.powlim.x)
        self.axes.yaxis.get_major_formatter().set_powerlimits(self.powlim.y)
        
    def apply_size(self):
        if(self.size!=None):
            plt.gcf().set_size_inches(self.size)
            
    def apply_plot_properties(self):
        self.apply_powlim()
        self.apply_size()
        self.axes.grid(b=True, which='major', axis='both')
        plt.axis(self.axis_style)
        if(self.ylim != None):
            self.axes.set_ylim(self.ylim)
        if(self.xlim != None):
            self.axes.set_xlim(self.xlim)
        if(self.additional_limits != None):
            xlim = list(self.axes.get_xlim())
            ylim = list(self.axes.get_ylim())

            xlim[0] -= self.additional_limits[0]
            xlim[1] += self.additional_limits[1]
            ylim[0] -= self.additional_limits[2]
            ylim[1] += self.additional_limits[3]

            self.axes.set_xlim(xlim)
            self.axes.set_ylim(ylim)
             
    def getNextLineStyle(self):
        self.line_style_pos += 1
        self.line_style_pos = self.line_style_pos%len(self.line_style)
        return self.line_style[self.line_style_pos]
        
    def resetLineStyle(self):
        self.line_style_pos = -1
        
    def getNextLineWidth(self):
        self.line_width_pos +=1
        self.line_width_pos = self.line_width_pos%len(self.line_width)
        return self.line_width[self.line_width_pos]
        
    def resetLineWidth(self):
        self.line_width_pos = -1
        
    def resetLineProperties(self):
        self.resetLineStyle()
        self.resetLineWidth()
        
    def plotPlot(self, trace):
        if(isinstance(trace,(list,tuple))):
            if(len(trace[0]) == 1):
                self.axes.plot(np.asarray(trace[0])*self.xmul, np.asarray(trace[1])*self.ymul, self.getNextLineStyle()[0]+'o', linewidth=self.getNextLineWidth(), markersize=self.markersize) #round marker instead of line
            else:
                self.axes.plot(np.asarray(trace[0])*self.xmul, np.asarray(trace[1])*self.ymul, self.getNextLineStyle(), linewidth=self.getNextLineWidth(), markersize=self.markersize)
        elif(isinstance(trace,dict)):
            if(trace['name'] == 'axvline'):
                if(not('kwdata' in trace)):
                    trace['kwdata'] = {}
                else:
                    if(not('color' in trace['kwdata'])):
                        trace['kwdata']['color'] = self.getNextLineStyle()[0]
                self.axes.axvline(*trace['data'], **trace['kwdata'])
            elif(trace['name'] == 'axhline'):
                if(not('kwdata' in trace)):
                    trace['kwdata'] = {}
                else:
                    if(not('color' in trace['kwdata'])):
                        trace['kwdata']['color'] = self.getNextLineStyle()[0]
                self.axes.axhline(*trace['data'], **trace['kwdata'])
            else:
                raise ValueError("object '%s' is not supported."%(trace['name']))
        else:
            raise TypeError("type: '%s' is not accepted by plotNicely."%(type(trace)))
            
        
    def plotAll(self):
        self.resetLineProperties()
        for data_i in self.data:
            self.plotPlot(data_i)
        self.apply_plot_properties()
        
    def plotCurrent(self):
        self.axes.clear()
        if(self.multiple_traces):
            self.resetLineProperties()
            for trace in self.data[self.current_plot]:
                self.plotPlot(trace)
        else:
            self.plotPlot(self.data[self.current_plot],0)
        self.apply_plot_properties()
        self.axes.set_title("Measure %d"%(self.current_plot))
        
    def showNextPlot(self, step):
        lims = xy_data(self.axes.get_xlim(), self.axes.get_ylim())
        self.current_plot += step
        if(self.current_plot >= len(self.data)):
            self.current_plot = 0
        elif(self.current_plot < 0):
            self.current_plot = len(self.data) - 1
        self.plotCurrent()
        self.axes.set_xlim(lims.x)
        self.axes.set_ylim(lims.y)
        self.figure.canvas.draw()
        
    def scrollPlots(self):
        if(self.scroll_step != 0):
            self.showNextPlot(self.scroll_step)
            Timer(3e-2, self.scrollPlots).start()
        
        
    def onKeyDownHandler(self, event):
        if( event.key == 'ctrl+up'):
            self.showNextPlot(1)
        elif( event.key == 'ctrl+down'):
            self.showNextPlot(-1)
        elif( event.key == 'ctrl+right'):
            self.scroll_step = 1
            self.scrollPlots()
        elif( event.key == 'ctrl+left'):
            self.scroll_step = -1
            self.scrollPlots()
#        lims = xy_data(self.axes.get_xlim(), self.axes.get_ylim())
#        if( event.key == 'ctrl+up'):
#            self.current_plot += 1
#            if(self.current_plot >= len(self.data)):
#                self.current_plot = 0
#            self.plotCurrent()
#        if( event.key == 'ctrl+down'):
#            self.current_plot -= 1
#            if(self.current_plot < 0):
#                self.current_plot = len(self.data) - 1
#            self.plotCurrent()
#        self.axes.set_xlim(lims.x)
#        self.axes.set_ylim(lims.y)
#        self.figure.canvas.draw()
        
    def onKeyUpHandler(self, event):
        self.scroll_step = 0
            
    def onMouseDownHandler(self, event):
        if(event.dblclick == True):
            #doubleclick -> reset zoom
            self.axes.axis(self.axis_style)
            self.figure.canvas.draw()
            return
        if( (event.button == 1) and (event.inaxes == self.axes) ):
            self.box_zooming.start(event)
            #print "starting box zoom at:", self.box_zooming.start_pos.x, self.box_zooming.start_pos.y
        if( (event.button == 2) and (event.inaxes == self.axes) ):
            self.panning.start(event)
        if( event.button == 3 ):
            self.box_zooming.abort()
            self.panning.abort()
            
    def onMouseUpHandler(self, event):
        self.box_zooming.end(event)
        self.panning.end(event)
        
    def onMouseMoveHandler(self, event):
        self.box_zooming.moving(event)
        self.panning.moving(event)
        
    def onMouseScrollHandler(self, event):
        base_scale = 0.8
        cur_xlim = self.axes.get_xlim()
        cur_ylim = self.axes.get_ylim()

        xdata = event.xdata # get event x location
        ydata = event.ydata # get event y location

        if event.button == 'down':
            # deal with zoom in
            scale_factor = 1 / base_scale
        elif event.button == 'up':
            # deal with zoom out
            scale_factor = base_scale
        else:
            # deal with something that should never happen
            scale_factor = 1
            print event.button

        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor

        relx = (cur_xlim[1] - xdata)/(cur_xlim[1] - cur_xlim[0])
        rely = (cur_ylim[1] - ydata)/(cur_ylim[1] - cur_ylim[0])

        self.axes.set_xlim([xdata - new_width * (1-relx), xdata + new_width * (relx)])
        self.axes.set_ylim([ydata - new_height * (1-rely), ydata + new_height * (rely)])
        self.figure.canvas.draw() # force re-draw

    
#==============================================================================
# for testing
#==============================================================================
if __name__ == "__main__":    
    plt.close('all')
    print "WARNING: this is main!!!"
    #import fileManagement as fM
    #path = r"F:\Studienarbeit\Messungen\Schall\Kegel\Verschiebung_vertikal__kegelspitze-empfaengermitte__kegel-kapsel-14mm"
    #measures = fM.readMatfiles(path)
    #a = plot([meas.getPlotDataA()[0] for meas in measures], scroll=True)
    
#    data = [ [([1,2,3],[1,1,1]), ([1,2,3],[0,1,2]), ([1.5],[1.5])],
#             (([1,2,3],[2,2,2]), ([1,2,3],[0,2,3]))  ]
#    vline = {'name':'axvline', 'data':[1.5,], 'kwdata':{'ls':'--', 'lw':3}}
#    data[0].append(vline)
#    a = plot(data, multiple_traces=True, scroll=True, additional_limits=[0,0,0.5,0.5])
    
    data1 = [ ([1,2,3],[0,1,2])]
    foo = plot(data1, line_style=[".-"])
    plt.xlabel("xlabel")
    plt.ylabel("left label")
    data2 = [ ([1,1.5,2],[100,200,370])]
    bar = plot(data2, right_axes=foo, line_style=["r.-"])
    plt.ylabel("right label")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    









