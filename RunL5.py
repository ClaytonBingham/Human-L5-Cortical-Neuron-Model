from CellManager import CellManager
from neuron import h
import os, sys
from multiprocessing import Process
from time import sleep
import numpy as np
import time
from scipy.spatial.distance import cdist
import pandas as pd
import random
import matplotlib.pyplot as plt
import pickle
from CostFunctions import SignalProcessing as sigproc

class NewCell():
	def __init__(self,pycell,types,lengths,diams,amp,plot=False,modify={}):
		self.cell = CellManager(pycell,types,lengths,diams)
		if modify != {}:
			self.cell.modify_somatodendritic_biophysics(modify)
		stim = self.stimulator(self.cell.soma[0],amp)
		data,labels = self.ez_record(h,var='v',sections=list(self.cell.soma))
		h.run()
		self.clean_data = self.ez_convert(data)
		sectypes = ['soma' for i in range(len(self.cell.soma))]
		if plot:
			x,y = self.plot_v_traces(self.clean_data,h.tstop,sectypes)
#		self.load_append_pickle('original_soma_voltage_traces.pickle',[item[0] for item in self.clean_data],amp)
		sp = sigproc()
		clay = sp.butter_lowpass_filter(self.clean_data.ravel(), cutoff=30, fs=64000, order=5)
		self.response = sp.mean_filtered_response(clay)
		self.spikes = len(sp.count_independent_crossings(self.clean_data))
		self.end_volt = self.clean_data[-1][0]
		for sec in h.allsec():
			h.delete_section(sec=sec)
	
	def ez_record(self,h,var='v',sections=None,order=None,targ_names=None,cust_labels=None,level='section'):
		"""
		Records state variables across segments

		Args:
			h = hocObject to interface with neuron
			var = string specifying state variable to be recorded.
				  Possible values are:
				      'v' (membrane potential)
				      'cai' (Ca concentration)
			sections = list of h.Section() objects to be recorded
			targ_names = list of section names to be recorded; alternative
				         passing list of h.Section() objects directly
				         through the "sections" argument above.
			cust_labels =  list of custom section labels
			level = 'section' or 'segment', determines if one or many positions are recorded for each section
		
		Returns:
			data = list of h.Vector() objects recording membrane potential
			labels = list of labels for each voltage trace
		"""
		data, labels = [], []
		for i in range(len(sections)):
			sec = sections[i]
			if level=='section':
				# record data
				data.append(h.Vector())
				if var == 'v':
					data[-1].record(sec(0.5)._ref_v)
				elif var == 'cai':
					data[-1].record(sec(0.5)._ref_cai)
				# determine labels
				if cust_labels is None:
					lab = sec.name()+'_'+str(round(0.5,5))
				else: 
					lab = cust_labels[i]+'_'+str(round(0.5,5))
				labels.append(lab)
			else:
				positions = np.linspace(0,1,sec.nseg+2)
				for position in positions[1:-1]:
					# record data
					data.append(h.Vector())
					if var == 'v':
						data[-1].record(sec(position)._ref_v)
					elif var == 'cai':
						data[-1].record(sec(position)._ref_cai)
					# determine labels
					if cust_labels is None:
						lab = sec.name()+'_'+str(round(position,5))
					else: 
						lab = cust_labels[i]+'_'+str(round(position,5))
					labels.append(lab)
		return (data, labels)

	def ez_convert(self,data):
		"""
		Takes data, a list of h.Vector() objects filled with data, and converts
		it into a 2d numpy array, data_clean. This should be used together with
		the ez_record command. 
		"""
		data_clean = np.empty((len(data[0]),len(data)))
		for (i,vec) in enumerate(data):
			data_clean[:,i] = vec.to_python()
		return(data_clean)

	def stimulator(self,sec,amp):
		stim = h.IClamp(sec(0.5))
		vec = h.Vector()
		netcon = h.NetCon(sec(0.5)._ref_v, None, sec=sec)
		netcon.record(vec)
		stim.dur = 1000
		stim.amp = amp
		stim.delay = 100
		return(stim,vec)

	def load_append_pickle(self,fname,data,label):
		import pickle
		try:
			with open(fname,'rb') as f:
				fdata = pickle.load(f)
			
			fdata.append(data)
			with open(fname,'wb') as f:
				pickle.dump(fdata,f)
		
		except:
			with open(fname,'wb') as f:
				fdata = []
				fdata.append(data)
				pickle.dump(fdata,f)

	def plot_v_traces(self,clean_data,timestop,labels):
		import matplotlib.pyplot as plt
		fig = plt.figure()
		plt.title('r-ais/hil/nd; g-pnd; k-internd; c-bt; m-interbt')
		colors = dict(zip(['soma','hillock','iseg','apical','basal','node','paranode1','paranode2','internode','bouton','interbouton','tuft'],['y','r','r','y','y','r','g','g','k','c','m','y']))
		for i in range(0,len(clean_data[0]),1)[::-1]:
			if labels[i] != 'soma':
				continue
			plt.plot(np.arange(0,timestop+h.dt,h.dt),[item[i] for item in clean_data],color=colors[labels[i]],label='Section '+str(i))
			plt.xlabel('ms')
			plt.ylabel('mV')
			plt.pause(0.005)
		
		plt.show()
		return(np.arange(0,timestop+h.dt,h.dt),[item[i] for item in clean_data])


def output_sptime_params(params,score):
	with open('parallelresults.txt','a') as f:
		f.write(str(params)+'\n'+str(score)+'\n')

def get_biophys(cellnum=0):
	biophys = dict(zip(['ra_soma','ra_basal','ra_apical','ra_hillock','cm_soma','cm_basal','cm_apical','cm_hillock','gpas_soma','gpas_basal','gpas_apical','gpas_hillock',
	'CaDynamics','gamma','gSKv3','gK','gSK','gIm','gIh','epas_soma','epas_basal','epas_apical','epas_hillock','ascm','gCaLVA','gCaHVA','gKTst','Nap',
	'NaTasoma','NaTaapical','decay','NaTaShift','NapShift','Kshift','SKv3Shift','aIh','bIh','cIh','dIh','eIh','kIh','vhIh','ehcnIh'],[float(arg) for arg in sys.argv[1:]]))

	if biophys == {}:
		with open(os.getcwd()+'/'+'MatingPool.pickle','rb') as f:
			mpool = pickle.load(f)
		cellnum = 0
		params = mpool[0][cellnum]
		print(cellnum)
		
		biophys = dict(zip(['ra_soma','ra_basal','ra_apical','ra_hillock','cm_soma','cm_basal','cm_apical','cm_hillock','gpas_soma','gpas_basal','gpas_apical','gpas_hillock',
	'CaDynamics','gamma','gSKv3','gK','gSK','gIm','gIh','epas_soma','epas_basal','epas_apical','epas_hillock','ascm','gCaLVA','gCaHVA','gKTst','Nap',
	'NaTasoma','NaTaapical','decay','NaTaShift','NapShift','Kshift','SKv3Shift','aIh','bIh','cIh','dIh','eIh','kIh','vhIh','ehcnIh'],params))
	return(biophys)

def MO_score(a,b,c):
	max_a = 350
	max_b = 161
	max_c = 15
	if a > max_a:
		a = max_a
	if b > max_b:
		b = max_b
	if c > max_c:
		c = max_c
	
	return((a/max_a + b/max_b + c/max_c)/3)

if __name__ == "__main__":
	run_default = False
	if not run_default:
		biophys = get_biophys()
	
	amplitudes = np.array([-110,-90,-50,-30,-10,10,30,50,70,90,110,130,160,170,190,210,230,250,270,310])
	amplitudes=amplitudes*0.001
	h.load_file("stdrun.hoc") #this loads standard simulation initialized variables
	stimamp = list(np.array([-100.0,20,0]))
	stimamp = list(np.array([1,0]))
	h.tstop = 1201
	h.celsius = 37
	h.v_init = -72
	h.finitialize(h.v_init)
	h.stdinit()
	h.steps_per_ms = 4
	h.dt = 0.25
	celldir = '/home/clayton/Desktop/Projects/CorticospinalTractModeling/2_FitBiophysics/L5Cell/Cell2/'
	pycell = celldir+[fname for fname in os.listdir(celldir) if '.py' in fname][0]
	types = celldir+[fname for fname in os.listdir(celldir) if 'section_labels' in fname][0]
	lengths = celldir+[fname for fname in os.listdir(celldir) if 'section_lengths' in fname][0]
	diams = celldir+[fname for fname in os.listdir(celldir) if 'section_diameters' in fname][0]
	response_list = []
	spike_counts = []
	end_volts = []
	sp = sigproc()
	for amp in amplitudes:
		cell = NewCell(pycell,types,lengths,diams,amp,plot=False,modify=biophys)
		response_list.append(cell.response)
		spike_counts.append(cell.spikes)
		end_volts.append(cell.end_volt)
		del(cell)

	
	cost = MO_score(sp.compare_to_reference_spikes(spike_counts),sp.compare_to_reference_floor_amplitude(response_list),sp.return_to_rest(end_volts))
	output_sptime_params(list(biophys.values()),cost)

