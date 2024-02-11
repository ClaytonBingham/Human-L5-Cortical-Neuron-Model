#from roots.microstructures import Microstructures
#from iterativemicrostructures import IterativeMicrostructures
from roots.root2neuron import Root2Py
from roots.swcToolkit import swcToolkit
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from collections import Counter
from scipy.spatial.distance import cdist
import copy

def write_section_centers(arbor,centers = [],target='sectionCenters.csv'):
	if centers == []:
		print('no centers passed, finding now and writing to '+target)
		for branch in arbor.keys():
			for section in arbor[branch]:
				centers.append(section[int(len(section)/2)])
	
	else:
		labelli = []
		for branch in centers.keys():
			for label in centers[branch]:
				labelli.append(label)
		
		centers=labelli
	
	df = pd.DataFrame()
	df['sectionCenters'] = range(len(centers))
	df['x'] = [center[0] for center in centers]
	df['y'] = [center[1] for center in centers]
	df['z'] = [center[2] for center in centers]
	df.to_csv(target,index=False)

def write_labels_to_csv(labels,target='sectionType.csv'):
	try:
		labelli = []
		for branch in labels.keys():
			for label in labels[branch]:
				labelli.append(label)
	except:
		labelli = labels
	
	print(len(labelli),'len labels')
	df = pd.DataFrame()
	df['sectionList'] = range(len(labelli)) #list(labelli.keys())
	df['sectionType'] = labelli #list(labelli.values())
	df.to_csv(target,index=False)

def write_lengths_to_csv(labels,target='sectionLengths.csv'):
	try:
		labelli = []
		for branch in labels.keys():
			for label in labels[branch]:
				labelli.append(label)
	except:
		labelli = labels
	
	print(len(labelli),'len lengths')
	df = pd.DataFrame()
	df['sectionList'] = range(len(labelli)) #list(labelli.keys())
	df['sectionLength'] = labelli #list(labelli.values())
	df.to_csv(target,index=False)

def write_diams_to_csv(labels,target='sectionDiameters.csv'):
	try:
		labelli = []
		for branch in labels.keys():
			for label in labels[branch]:
				labelli.append(label)
	except:
		labelli = labels
	
	print(len(labelli),'len diams')
	df = pd.DataFrame()
	df['sectionList'] = range(len(labelli)) #list(labelli.keys())
	df['sectionLength'] = labelli #list(labelli.values())
	df.to_csv(target,index=False)

def eucdist3d(point1,point2):
	"""
	
	euclidean distance between point1 and point2 - [x,y,z]
	
	"""
	
	return(((point2[0]-point1[0])**2 + (point2[1]-point1[1])**2 + (point2[2]-point1[2])**2)**0.5)

def labels_from_swc(fname,morph,simplified=True):
	labels = []
	points = []
	with open(fname,'r') as f:
		rows = f.readlines()
	
	types = dict(zip([1,2,3,4,5],['soma','hillock','basal','apical','other']))
	for row in rows:
		row = row.split(' ')
		points.append((float(row[2]),float(row[3]),float(row[4])))
		labels.append(types[int(row[1])])
	
	branchedlabels = {}
	for branch in morph.keys():
		branchedlabels[branch] = []
		for point in morph[branch]:
			branchedlabels[branch].append(labels[points.index(point[:3])])
		
		if simplified:
			branchedlabels[branch] = [branchedlabels[branch][-1]]
	
	return(branchedlabels)

def lengths_from_morph(morph):
	branchedlengths = {}
	for branch in morph.keys():
		branchedlengths[branch] = 0
		for p,point in enumerate(morph[branch][:-1]):
			branchedlengths[branch]+=eucdist3d(morph[branch][p],morph[branch][p+1])
		branchedlengths[branch] = [branchedlengths[branch]]
	
	return(branchedlengths)

def fitted_curve(stream,streamlen,interval):
	if streamlen < interval*2 or len(stream) < 4:
		return([stream[0][:3],stream[-1][:3]])
	
	tck, u = interpolate.splprep([[p[0] for p in stream],[p[1] for p in stream],[p[2] for p in stream]], s=2)
	u_fine = np.linspace(0,1,int(streamlen/interval))
	x, y, z = interpolate.splev(u_fine, tck)
	newstream = [[x[i],y[i],z[i]] for i in range(len(x))]
	newstream[0] = stream[0][:3]
	newstream[-1] = stream[-1][:3]
	return(newstream)

def resegment_morph(fname,morph,lengths,interval):
	newmorph = copy.deepcopy(morph)
	labels = labels_from_swc(fname,morph,simplified=False)
	for branch in morph.keys():
		x = [(morph[branch][p][-1]+morph[branch][p+1][-1])/2.0 for p in range(len(morph[branch]))[:-1]]
		y = lengths[branch]
		first = morph[branch][0][0]
		last = morph[branch][0][-1]
		newmorph[branch] = fitted_curve(morph[branch],lengths[branch][0],interval)
		newmorph[branch] = [list(item)+return_nearest_diam_type(item,morph[branch],labels[branch]) for item in newmorph[branch]]
	return(newmorph)

def most_common(lst):
	data = Counter(lst)
	return(data.most_common(1)[0][0])

def return_nearest_diam_type(point,li,types):
	nearest = np.argmin(cdist([point],[l[:3] for l in li]))
	return([li[nearest][3],types[nearest]])

def weighted_average(x,y):
	return(np.sum([x[i]*y[i] for i in range(len(x))])/np.sum(y))

def morph_radius_to_diameter(morph):
	for branch in morph.keys():
		for p,point in enumerate(morph[branch]):
			morph[branch][p] = (morph[branch][p][0],morph[branch][p][1],morph[branch][p][2],morph[branch][p][3]*2.0)
	
	return(morph)

def prepend_soma_to_morph(morph):
	newmorph = {}
	newmorph[0] = [morph[0][0],morph[0][0]]
	for branch in morph.keys():
		newmorph[len(newmorph.keys())] = morph[branch]
	
	return(newmorph)

def branches_to_sections(morph):
	for branch in morph.keys():
		morph[branch] = [[list(morph[branch][i]),list(morph[branch][i+1])] for i in range(len(morph[branch]))[:-1]]
	
	return(morph)

def strip_return_labels_diams_lengths(morph):
	labels = []
	diams = []
	lengths = []
	for branch in morph.keys():
		for s,section in enumerate(morph[branch]):
			labels.append(section[1][-1])
			diams.append((section[0][3]+section[1][3])/2.0)
			lengths.append(eucdist3d(section[0][:3],section[1][:3]))
			morph[branch][s] = [section[0][:4],section[1][:4]]
	
	return(morph,diams,lengths,labels)

if __name__ == "__main__":
	swctool = swcToolkit()
	py_writer = Root2Py()
	swc = '/home/clayton/Desktop/Projects/CorticospinalTractModeling/2_FitBiophysics/L5Cell/Cell2/H16.06.012.11.06.02_679883747_m.swc'
	morph = swctool.load(swc)
	morph = morph_radius_to_diameter(morph)
	morph = prepend_soma_to_morph(morph)
	labels_ = labels_from_swc(swc,morph)
	lengths_ = lengths_from_morph(morph)
	clay = resegment_morph(swc,morph,lengths_,6.75)
	clay = branches_to_sections(clay)
	clay,diams_,lengths_,labels_ = strip_return_labels_diams_lengths(clay)
	soma_sphere2cyl_length = 14.1855
	lengths_[0] = soma_sphere2cyl_length
	py_writer.arbor_to_nrn(clay,labels_,target=swc.strip('.swc')+'.py',suppress_points=True)
	write_labels_to_csv(labels_,target=swc.strip('.swc')+'_section_labels'+'.csv')
	write_lengths_to_csv(lengths_,target=swc.strip('.swc')+'_section_lengths'+'.csv')
	write_section_centers(clay,target=swc.strip('.swc')+'_section_centers'+'.csv')
	write_diams_to_csv(diams_,target=swc.strip('.swc')+'_section_diameters'+'.csv')
	

