import numpy as np
from copy import deepcopy

class LinearAddMicrostructures():
	def __init__(self,arbor,myelinlist=[],boutonlist=[],myelindimensions=[['node','paranode1','paranode2','internode','paranode2','paranode1'],[1,3,35,234,35,3]],boutondimensions=[['bouton','interbouton'],[5,28]]):
		self.arbor = deepcopy(arbor)
		self.remove_diameters()
		self.labels = dict(zip(self.arbor.keys(),[[] for key in self.arbor.keys()]))
		for m in myelinlist:
			self.arbor[m],self.labels[m] = self.iterate_microstructures_on_branch(self.arbor[m],myelindimensions[1],myelindimensions[0])
		
		for b in boutonlist:
			self.arbor[b],self.labels[b] = self.iterate_microstructures_on_branch(self.arbor[b],boutondimensions[1],boutondimensions[0])
	
		self.get_arbor_lengths_and_centers()
		self.add_back_dummy_diameters()
		self.find_remove_zero_len_sections()
	
	def remove_diameters(self):
		for branch in self.arbor:
			self.arbor[branch] = [point[:3] for point in self.arbor[branch]]
	
	
	def check_return_results(self):
		if len(self.arbor.values()) != len(self.labels.values()) != len(self.lengths.values()):
			print("Arbor, Labels, and Lengths aren't lined up!...double check what you are getting")
		return(self.arbor,self.labels,self.lengths,self.centers)
	
	def get_vector_and_magnitude(self,a,b):
		nb = np.array(b)
		na = np.array(a)
		vec = nb-na
		return(vec,np.linalg.norm(vec))

	def try_get_section_from_vec(self,a,b,seclen):
		vec,mag = self.get_vector_and_magnitude(a,b)
		if mag >= seclen:
			return(list(np.array(a)+vec*(seclen/mag)),seclen)
		if mag < seclen:
			return(b,mag)

	def eucdist3d(self,point1,point2):
		"""
		
		euclidean distance between point1 and point2 - [x,y,z]
		
		"""
		
		return(((point2[0]-point1[0])**2 + (point2[1]-point1[1])**2 + (point2[2]-point1[2])**2)**0.5)
	
	def find_remove_zero_len_sections(self):
		for branch in self.lengths.keys():
			for s,section in enumerate(self.lengths[branch]):
				if round(section,2) == 0:
					del(self.lengths[branch][s])
					del(self.arbor[branch][s])
					del(self.centers[branch][s])
					del(self.labels[branch][s])
					print('deleted 0 length '+self.labels[branch][s]+' from branch '+str(branch)+' section '+str(s))
	
	def get_section_length(self,section):
		return(np.sum([self.eucdist3d(section[p],section[p+1]) for p in range(len(secton))]))
	
	def find_insert_center(self,section,length):
		section = [list(item) for item in section]
		if len(section) == 2:
			newsection = [section[0],self.midpoint(section[0],section[1]),section[1]]
			center = newsection[1]
			return(newsection,center)
		
		else:
			seglens = [self.eucdist3d(section[i],section[i+1]) for i in range(len(section[:-1]))]
			for s,seg in enumerate(seglens):
				if np.sum(seglens[:s+1]) > length/2.0:
					break
			
			remainder = np.abs(length/2.0-np.sum(seglens[:s]))
			center,mag = self.try_get_section_from_vec(section[s],section[s+1],remainder)
#			print(section[:s-2]+[center]+section[s-2:],[center],remainder==mag,length/2.0,np.sum(seglens),self.eucdist3d(center,section[0]),self.eucdist3d(center,section[1]),'get offset center')
			return(section[:s-2]+[center]+section[s-2:],center)
	
	def add_back_dummy_diameters(self):
		for branch in self.arbor.keys():
			for s,section in enumerate(self.arbor[branch]):
				for p,point in enumerate(section):
					self.arbor[branch][s][p] = list(self.arbor[branch][s][p])+[1]
	
	
	def get_arbor_lengths_and_centers(self):
		self.lengths = {}
		self.centers = {}
		for branch in self.arbor.keys():
			self.lengths[branch] = []
			self.centers[branch] = []
			for i,item in enumerate(self.arbor[branch]):
				self.lengths[branch].append(0)
				for p,point in enumerate(item[:-1]):
					self.lengths[branch][-1]+=self.eucdist3d(point,item[p+1])
				
				newsection,center = self.find_insert_center(item,self.lengths[branch][-1])
				self.arbor[branch][i] = newsection
				self.centers[branch].append(center)
				
				
	
	def midpoint(self,a,b):
		try:
			midpoint = [item/2.0 for item in [a[0]+b[0],a[1]+b[1],a[2]+b[2],a[3]+b[3]]]
		except:
			midpoint = [item/2.0 for item in [a[0]+b[0],a[1]+b[1],a[2]+b[2]]]
		return(midpoint)
	
	def try_get_section_from_branch(self,branch,seclen):
		down_branch = deepcopy(branch)
		newsection = [down_branch.pop(0)]
		while seclen > 0 and len(down_branch) > 0:
			end,secless = self.try_get_section_from_vec(newsection[-1],down_branch[0],seclen)
			if down_branch[0] == end:
				down_branch.pop(0)
			
			newsection.append(end)
			seclen-=secless
		
		down_branch = [end]+down_branch
		return(newsection,down_branch)

	def reserve_last_section(self,down_branch,seclen):
		localbranch = list(reversed(down_branch))
		lastsection,localbranch = self.try_get_section_from_branch(localbranch,seclen)
		return(list(reversed(lastsection)),list(reversed(localbranch)))

	def iterate_microstructures_on_branch(self,branch,microstructure_dimensions,microstructures_labels):
		dimind = 0
		newbranch = []
		newlabels = []
		localbranch = deepcopy(branch)
		lastsection,remaining = self.reserve_last_section(localbranch,microstructure_dimensions[dimind])
		lastlabel = microstructures_labels[dimind]
		while len(remaining) >1:
			newsection,remaining = self.try_get_section_from_branch(remaining,microstructure_dimensions[dimind])
			newbranch.append(newsection)
			newlabels.append(microstructures_labels[dimind])
			dimind+=1
			if dimind == len(microstructure_dimensions):
				dimind = 0
			
#			print(remaining)
		
		newbranch.append(lastsection)
		newlabels.append(lastlabel)
		return(newbranch,newlabels)




if __name__ == "__main__":
	from copy import deepcopy
	arbor = dict(zip([0],[[(i*1000,0,0) for i in range(11)]]))
	m = LinearAddMicrostructures(arbor,[0],[])
	newarbor,labels,lengths = m.check_return_results()

