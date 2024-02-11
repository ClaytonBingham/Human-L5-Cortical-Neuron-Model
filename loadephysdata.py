import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate as interp
import pickle

def allkeys(obj):
	"Recursively find all keys in an h5py.Group."
	keys = (obj.name,)
	if isinstance(obj, h5py.Group):
		for key, value in obj.items():
			if isinstance(value, h5py.Group):
				keys = keys + allkeys(value)
			else:
				keys = keys + (value.name,)
	return(keys)

def print_key(f,key):
	key = key.split('/')
	if len(key) == 1:
		print(key,str(f[key[0]]))
	if len(key) == 2:
		print(key,str(f[key[0]][key[1]]))
	if len(key) == 3:
		print(key,str(f[key[0]][key[1]][key[2]]))
	if len(key) == 4:
		print(key,str(f[key[0]][key[1]][key[2]][key[3]]))
	if len(key) == 5:
		print(key,str(f[key[0]][key[1]][key[2]][key[3]][key[4]]))
	if len(key) == 6:
		print(key,str(f[key[0]][key[1]][key[2]][key[3]][key[4]][key[5]]))

def upsample(oldy,oldx,newx):
	f = interp.interp1d(oldx,oldy,fill_value='extrapolate')
	return(f(newx))

def load_sim_result():
	x = np.arange(0,1300+0.015625,0.015625)
	with open('original_soma_voltage_traces.pickle','rb') as f:
		data = pickle.load(f)
	return(x,data)

if __name__ == "__main__":
	
	#load experimental results
	interval = 1/50000.0 *1000.0
	sim_interval = 0.015625
	f = h5py.File('/home/clayton/Desktop/Projects/CorticospinalTractModeling/2_FitBiophysics/L5Cell/Cell2/548421747_ephys.nwb','r')
	keys = allkeys(f)
	suptitle = 'Amplitude: '+str(f['epochs']['Sweep_65']['response']['timeseries']['aibs_stimulus_amplitude_pa'])+' Interval: '+str(f['epochs']['Sweep_65']['response']['timeseries']['aibs_stimulus_interval'])+' #Samples: '+str(f['epochs']['Sweep_65']['response']['timeseries']['num_samples'])+' Starting Time: '+str(f['epochs']['Sweep_65']['response']['timeseries']['starting_time'])
#	oldx = np.arange(0,1200,0.02)
#	newx = np.arange(0,1200,sim_interval)
	labels = [-110,-90,-50,-30,-10,10,30,50,70,90,110,130,160,170,190,210,230,250,270,310]
	upsampled_data = {}
	for i,key in enumerate([51,52,54,55,56,57,58,59,60,61,62,63,73,75,66,67,68,69,70,72]):
		stim = list(f['epochs']['Sweep_'+str(key)]['stimulus']['timeseries']['data'])[46000:106000]
		response = list(f['epochs']['Sweep_'+str(key)]['response']['timeseries']['data'])[46000:106000]
#		newstim = upsample(stim,oldx,newx)
#		newresponse = upsample(response,oldx,newx)
#		upsampled_data[i] = [labels[i],newx,newstim,newresponse]
		upsampled_data[i] = [labels[i],np.arange(0,1200,0.02),stim,response]
	
	from CostFunctions import SignalProcessing as sigproc
	sp = sigproc()
	levels = []
	for key in upsampled_data.keys():
		unfiltered = upsampled_data[key][3].ravel()
		unfiltered = sp.butter_lowpass_filter(unfiltered,cutoff=30,fs=64000,order=5)
		levels.append(sp.mean_filtered_response(unfiltered))
	fix,ax = plt.subplots(3)
	for key in upsampled_data.keys():
		lab,x,y0,y1 = upsampled_data[key]
		ax[0].plot(x,y0*1000.0,label=str(lab)+'pA')
		ax[1].plot(x,y1*1000.0)
	
	ax[0].legend(loc='upper center',bbox_to_anchor=(0.5,1.45),ncol=5,fancybox=True)
#	plt.show()
#	fig,ax = plt.subplots(2)
#	ax[0].plot(oldx,stim)
#	ax[0].plot(newx,newstim)
#	ax[1].plot(oldx,response)
#	ax[1].plot(newx,newresponse)
#	fig.suptitle(suptitle)
#	plt.show()

	#load simulation results
	x,data = load_sim_result()
	sim_labels = np.sort(labels)
	sim_labels = sim_labels*0.001
	
	for l,lab in enumerate(sim_labels):
		ax[2].plot(list(x)[:76800],data[l][:76800],label=str(1000*lab)+'pA')
	
#	plt.legend()
	plt.show()

