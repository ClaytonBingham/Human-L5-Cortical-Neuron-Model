from neuron import h
import numpy as np
import random

class CellManager():
	"""
	This class is used to assign biophysics and discretize NEURON morphologies after they are already created in NEURON python namespace.
	
	Example usage:
	
	```
	cm = CellManager()
	for section in sectionList:
		secref = cm.assign_biophysics_to_section(sec,type='node')
	
	cm.fixnseg()
	```
	
	This class is dependent upon having the fixnseg.hoc file in root and having nrnivmodl'd an AXNODE.mod and PARAK75.mod in /x86_64 in root.
	
	These items can be found in these places, respectively:
	
	https://senselab.med.yale.edu/ModelDB/ShowModel?model=114685&file=/JohnsonMcIntyre2008/GPe_model/#tabs-2
	
	https://www.neuron.yale.edu/neuron/static/docs/d_lambda/d_lambda.html

	"""
	
	def __init__(self,sectionlist_fname,types_fname,lengths_fname,diams_fname,CCF_outer=10.25):
		self.all = []
		self.soma = []
		self.hillock = []
		self.iseg = []
		self.basal = []
		self.apical = []
		self.tuft = []
		self.node = []
		self.paranode1 = []
		self.paranode2 = []
		self.internode = []
		self.interbouton = []
		self.bouton = []
		self.outerdiam = CCF_outer
		self.set_global_params()
		self.sort_section_types(self.get_sectionlist(sectionlist_fname),types_fname,lengths_fname,diams_fname)
		self.apply_biophysics()
		self.set_other_mod_globals() 
#		self.print_passive_properties_by_section_type()
	
	def get_sectionlist(self,sectionlist_fname):
		namespace = {}
		with open(sectionlist_fname,'r') as f:
			exec(f.read(),namespace)
		
		return(namespace['sectionList'])
	
	def set_global_params(self):
		self.axosomatic_cm = 1.5
		diams,lengths,self.nl = self.calculate_morph_vars(self.outerdiam)
		self.nodeD = diams[0]
		self.para1D = diams[1]
		self.para2D = diams[1]
		self.interD = diams[1]
		self.PARAM_decay=1000
		self.PARAM_Ra= 501.6
		self.PARAM_cm= 1.6226
		self.PARAM_e_pas= 84.325
		self.PARAM_g_pas= 1.75e-05
		self.PARAM_gCa_LVAstbar=0.00099587
		self.PARAM_gCa_HVAbar=1.7838e-09
		self.PARAM_gIhbar=5.0723e-05
		self.PARAM_gImbar=.0002
		self.PARAM_gK_Pstbar=0.07
		self.PARAM_gK_Tstbar=2e-05
		self.PARAM_gSK_E2bar=2.0964e-09
		self.PARAM_gSKv3_1bar=0.04
		self.PARAM_Nap_Et2=1e-06
		self.PARAM_NaTa_t_soma=2.2
		self.PARAM_NaTa_t_apical=0.001
		h.shift_NaTa_t=5
		h.shift_Nap_Et2=0
		h.shift_K_Pst=0
		h.shift_SKv3_1=10
		h.a_Ih=23.428
		h.b_Ih=0.21756
		h.c_Ih=1.3881e-09
		h.d_Ih=0.082329
		h.e_Ih=1.9419e-09
		h.k_Ih=8.0775
		h.vh_Ih=-90.963
		h.ehcn_Ih=-49.765
	
	def set_other_mod_globals(self):
		h.vtraub_axnode = -80
	
	def apply_biophysics(self):
		self.sectionreferences = []
		for s,sec in enumerate(self.soma):
			self.sectionreferences.append(self.assign_biophysics_to_section(sec,type='soma',size=self.sizes['soma'][s],diam=self.diams['soma'][s]))
		for s,sec in enumerate(self.basal):
			self.sectionreferences.append(self.assign_biophysics_to_section(sec,type='basal',size=self.sizes['basal'][s],diam=self.diams['basal'][s]))
		for s,sec in enumerate(self.apical):
			self.sectionreferences.append(self.assign_biophysics_to_section(sec,type='apical',size=self.sizes['apical'][s],diam=self.diams['apical'][s]))
		for s,sec in enumerate(self.tuft):
			self.sectionreferences.append(self.assign_biophysics_to_section(sec,type='tuft',size=self.sizes['tuft'][s],diam=self.diams['tuft'][s]))
		for s,sec in enumerate(self.hillock):
			self.sectionreferences.append(self.assign_biophysics_to_section(sec,type='hillock',size=self.sizes['hillock'][s],diam=self.diams['hillock'][s]))
		for s,sec in enumerate(self.iseg):
			self.sectionreferences.append(self.assign_biophysics_to_section(sec,type='iseg',size=self.sizes['iseg'][s],diam=self.diams['iseg'][s]))
		for s,sec in enumerate(self.node):
			self.sectionreferences.append(self.assign_biophysics_to_section(sec,type='node',size=self.sizes['node'][s]))
		for s,sec in enumerate(self.paranode1):
			self.sectionreferences.append(self.assign_biophysics_to_section(sec,type='paranode1',size=self.sizes['paranode1'][s]))
		for s,sec in enumerate(self.paranode2):
			self.sectionreferences.append(self.assign_biophysics_to_section(sec,type='paranode2',size=self.sizes['paranode2'][s]))
		for s,sec in enumerate(self.internode):
			self.sectionreferences.append(self.assign_biophysics_to_section(sec,type='internode',size=self.sizes['internode'][s]))
		for s,sec in enumerate(self.bouton):
			self.sectionreferences.append(self.assign_biophysics_to_section(sec,type='bouton',size=self.sizes['bouton'][s]))
		for s,sec in enumerate(self.interbouton):
			self.sectionreferences.append(self.assign_biophysics_to_section(sec,type='interbouton',size=self.sizes['interbouton'][s]))
	
	def sort_section_types(self,sectionlist,types_fname,lengths_fname,diams_fname):
		section_types = self.load_section_types(types_fname)
		section_lengths = self.load_section_lengths(lengths_fname)
		section_diams = self.load_section_diams(diams_fname)
		self.sizes = dict(zip(['soma','hillock','iseg','basal','apical','tuft','node','paranode1','paranode2','internode','bouton','interbouton'],[[],[],[],[],[],[],[],[],[],[],[],[]]))
		self.diams = dict(zip(['soma','hillock','iseg','basal','apical','tuft','node','paranode1','paranode2','internode','bouton','interbouton'],[[],[],[],[],[],[],[],[],[],[],[],[]]))
		dels = []
		for s,sec in enumerate(section_types):
			self.all.append(sectionlist[s])
			if sec == 'soma':
				self.soma.append(sectionlist[s])
				self.sizes[sec].append(section_lengths[s])
				self.diams[sec].append(section_diams[s])
			if sec == 'hillock':
				self.hillock.append(sectionlist[s])
				self.sizes[sec].append(section_lengths[s])
				self.diams[sec].append(section_diams[s])
			if sec == 'iseg':
				self.iseg.append(sectionlist[s])
				self.sizes[sec].append(section_lengths[s])
				self.diams[sec].append(section_diams[s])
			if sec == 'basal':
				self.basal.append(sectionlist[s])
				self.sizes[sec].append(section_lengths[s])
				self.diams[sec].append(section_diams[s])
			if sec == 'apical':
				self.apical.append(sectionlist[s])
				self.sizes[sec].append(section_lengths[s])
				self.diams[sec].append(section_diams[s])
			if sec == 'tuft':
				self.tuft.append(sectionlist[s])
				self.sizes[sec].append(section_lengths[s])
				self.diams[sec].append(section_diams[s])
			if sec == 'node':
				self.node.append(sectionlist[s])
				self.sizes[sec].append(section_lengths[s])
				self.diams[sec].append(section_diams[s])
			if sec == 'paranode1':
				self.paranode1.append(sectionlist[s])
				self.sizes[sec].append(section_lengths[s])
				self.diams[sec].append(section_diams[s])
			if sec == 'paranode2':
				self.paranode2.append(sectionlist[s])
				self.sizes[sec].append(section_lengths[s])
				self.diams[sec].append(section_diams[s])
			if sec == 'internode':
				self.internode.append(sectionlist[s])
				self.sizes[sec].append(section_lengths[s])
				self.diams[sec].append(section_diams[s])
			if sec == 'bouton':
				self.bouton.append(sectionlist[s])
				self.sizes[sec].append(section_lengths[s])
				self.diams[sec].append(section_diams[s])
			if sec == 'interbouton':
				self.bouton.append(sectionlist[s])
				self.sizes['bouton'].append(section_lengths[s])			
				self.diams[sec].append(section_diams[s])
	
	def load_section_types(self,fname):
		with open(fname,'r') as f:
			rows = f.readlines()
		
		return([item.strip().split(',')[-1] for item in rows[1:]])
	
	def load_section_lengths(self,fname):
		with open(fname,'r') as f:
			rows = f.readlines()
		
		return([float(item.strip().split(',')[-1]) for item in rows[1:]])
	
	def load_section_diams(self,fname):
		with open(fname,'r') as f:
			rows = f.readlines()
		
		return([float(item.strip().split(',')[-1]) for item in rows[1:]])
	
	def pretty_print_properties(self,section,sec_type):
		print('\nSection Type: '+sec_type)
		print('Section Ra  : '+str(section.Ra)+' Ohm-cm')
		print('Section Cm  : '+str(section.cm)+' uF/cm^2')
		print('Section diam: '+str(section.diam)+' um')
		print('Section Len : '+str(section.L)+' um')
#		print('Section xg  : '+str(section.xg[0]))
#		print('Section xc  : '+str(section.xc[0])+' uF/cm^2\n')
	
	def print_passive_properties_by_section_type(self):
		if self.soma != []:
			self.pretty_print_properties(self.soma[0],'Soma')
		if self.hillock != []:
			self.pretty_print_properties(self.hillock[0],'Hillock')
		if self.iseg != []:
			self.pretty_print_properties(self.iseg[0],'Initial Segment')
		if self.node != []:
			self.pretty_print_properties(self.node[0],'Node of Ranvier')
		if self.paranode1 != []:
			self.pretty_print_properties(self.paranode1[0],'1st Paranode')
		if self.paranode2 != []:
			self.pretty_print_properties(self.paranode2[0],'2nd Paranode')
		if self.internode != []:
			self.pretty_print_properties(self.internode[0],'Internode')
		if self.bouton != []:
			self.pretty_print_properties(self.bouton[0],'Bouton')
		if self.interbouton != []:
			self.pretty_print_properties(self.interbouton[0],'Interbouton')
		if self.basal != []:
			self.pretty_print_properties(self.basal[0],'Basal Dendrite')
		if self.tuft != []:
			self.pretty_print_properties(self.tuft[0],'Tuft Dendrite')
		if self.apical != []:
			self.pretty_print_properties(self.apical[0],'Apical Dendrite')
	
	def interpolate_fiber_dep_vars(self,fiberD,outerdiams):
		for d in range(len(outerdiams[:-1])):
			if fiberD>= outerdiams[d] and fiberD<=outerdiams[d+1]:
				return(d,d+1,float((fiberD-outerdiams[d])/(outerdiams[d+1]-outerdiams[d])))
		
		return(None)
	
	def calculate_new_dep_var(self,a,b,prop):
		return(a+(b-a)*prop)
	
	def get_values_from_fitted_curve(self,xreal,yreal,xnew):
		z = np.polyfit(xreal,yreal,2)
		f = np.poly1d(z)
		ynew = f(xnew)
		return(ynew)

	def dependent_vars(self,fiberD):
		ddict = {}
		ddict['outerdiams'] = [5.7,7.3,8.7,10.0,11.5,12.8,14.0,15.0,16.0]
		ddict['gs'] = [0.605,0.630,0.661,0.690,0.700,0.719,0.739,0.767,0.791]
		ddict['axonDs'] = [3.4,4.6,5.8,6.9,8.1,9.2,10.4,11.5,12.7]
		ddict['nodeDs'] = [1.9,2.4,2.8,3.3,3.7,4.2,4.7,5.0,5.5]
		ddict['paraD1s']=[1.9,2.4,2.8,3.3,3.7,4.2,4.7,5.0,5.5]
		ddict['paraD2s']=[3.4,4.6,5.8,6.9,8.1,9.2,10.4,11.5,12.7]
		ddict['deltaxs']=np.array([500,750,1000,1150,1250,1350,1400,1450,1500])
		ddict['paralength2s']=np.array([35,39,40,46,50,54,56,58,60])
		ddict['nls'] = [80,100,110,120,130,135,140,145,150]
		if fiberD < ddict['outerdiams'][0] or fiberD > ddict['outerdiams'][-1]:
			prop = None
		else:
			prop = self.interpolate_fiber_dep_vars(fiberD,ddict['outerdiams'])
		if prop is None:
			dep_vars = []
			for key in ddict.keys():
				if key == 'outerdiams':
					dep_vars.append(fiberD)
				else:
					dep_vars.append(self.get_values_from_fitted_curve(ddict['outerdiams'],ddict[key],fiberD))
			return(dep_vars)
		
		else:
			dep_vars = []
			for key in ddict.keys():
				dep_vars.append(self.calculate_new_dep_var(ddict[key][prop[0]],ddict[key][prop[1]],prop[2]))
		
		return(dep_vars)
	
	def calculate_morph_vars(self,fiberD):
		dep_vars = self.dependent_vars(fiberD)
		interlen = int((fiberD*65)-1-(2*3)-(2*dep_vars[7]))
		return([dep_vars[3],dep_vars[4],dep_vars[5],dep_vars[2],dep_vars[5],dep_vars[4]],[1,3,int(dep_vars[7]),interlen,int(dep_vars[7]),3],int(dep_vars[8]))
	
	def assign_biophysics_to_section(self,sec,rhoa_=0.7e6,type='node',size=1,diam=1):
		"""
		This class method takes a NEURON section, with an empirically validated rhoa and assigns McIntyre, Richardson, & Grill (2002) biophysics according to section type (including types: 'node','paranode1','paranode2','internode','unmyelinated') and according to Bahl et al. for ('soma','basal','apical','tuft)
		
		"""
		if type == 'hillock':
			secref = self.assign_hillock_biophysics(sec,rhoa_,0.002,sectionlen=size,section_diam=diam)
#			secref = self.assign_nodal_biophysics(sec,rhoa_,0.002,self.outerdiam,self.nodeD,sectionlen=None)
		if type == 'iseg':
			secref = self.assign_initial_segment_biophysics(sec,rhoa_,0.002,sectionlen=size,section_diam=diam)
		if type=='node':
			secref = self.assign_nodal_biophysics(sec,rhoa_,0.002,self.outerdiam,self.nodeD,sectionlen=size)
		if type=='paranode1':
			secref = self.assign_paranode1_biophysics(sec,rhoa_,0.002,self.outerdiam,self.para1D,self.nl,sectionlen=size)
		if type=='paranode2':
			secref = self.assign_paranode2_biophysics(sec,rhoa_,0.004,self.outerdiam,self.para2D,self.nl,sectionlen=size)
		if type=='internode':
			secref = self.assign_internode_biophysics(sec,rhoa_,0.004,self.outerdiam,self.interD,self.nl,sectionlen=size)
		if type=='bouton':
			secref = self.assign_bouton_biophysics(sec,rhoa_,0.002,sectionlen=size)
		if type=='interbouton':
			secref = self.assign_interbouton_biophysics(sec,rhoa_,0.002,sectionlen=size)
		if type=='soma':
			secref = self.assign_soma_biophysics(sec,rhoa_,0.002,sectionlen=size,section_diam=diam)
		if type=='basal':
			secref = self.assign_basal_biophysics(sec,rhoa_,0.002,sectionlen=size,section_diam=diam)
		if type=='apical':
			secref = self.assign_apical_biophysics(sec,rhoa_,0.002,sectionlen=size,section_diam=diam)
		if type=='tuft':
			secref = self.assign_tuft_biophysics(sec,rhoa_,0.002,sectionlen=size,section_diam=diam)
		
		h.pop_section()
		return(secref)
	
	def fixnseg(self):
		h.xopen("fixnseg.hoc")
		h.geom_nseg()
	
	def lambda_f(self,sec):
		return((100000)*(sec.diam/(4*np.pi*100*sec.Ra*sec.cm))**0.5)
	
	def dlambda(self,sec):
		return(int((sec.L/(0.1*self.lambda_f(sec)+0.9))/2)*2+1)
	
	def calc_ra(self,diameter):
		return(5497.787143782139*(1/(np.pi*(diameter/2)**2)))
	
	def assign_soma_biophysics(self,sec,rhoa,space_p,sectionlen,section_diam):
		sec.push()
		secref = h.SectionRef()
		if sectionlen is not None:
			sec.L = sectionlen
		if section_diam is not None:
			sec.diam = section_diam
		sec.Ra = self.PARAM_Ra
		sec.cm = self.axosomatic_cm 
		sec.nseg = self.dlambda(sec)
		sec.insert('pas')
		sec.g_pas=self.PARAM_g_pas
		sec.e_pas=-1*self.PARAM_e_pas
		sec.insert('Ca_LVAst')
		sec.insert('Ca_HVA')
		sec.insert('SKv3_1')
		sec.insert('K_Pst')
		sec.insert('K_Tst')
		sec.insert('SK_E2')
		sec.insert('CaDynamics_E2')
		sec.insert('Ih')
		sec.insert('Nap_Et2')
		sec.insert('NaTa_t')
		sec.gCa_LVAstbar_Ca_LVAst = self.PARAM_gCa_LVAstbar
		sec.gCa_HVAbar_Ca_HVA = self.PARAM_gCa_HVAbar
		sec.decay_CaDynamics_E2 = 460.0
		sec.gamma_CaDynamics_E2 = 0.000501
		sec.gSKv3_1bar_SKv3_1 = self.PARAM_gSKv3_1bar
		sec.gK_Pstbar_K_Pst = self.PARAM_gK_Pstbar
		sec.gK_Tstbar_K_Tst = self.PARAM_gK_Tstbar
		sec.gSK_E2bar_SK_E2 = self.PARAM_gSK_E2bar
		sec.gIhbar_Ih = self.PARAM_gIhbar
		sec.gNap_Et2bar_Nap_Et2 = self.PARAM_Nap_Et2
		sec.gNaTa_tbar_NaTa_t = self.PARAM_NaTa_t_soma
#		sec.insert('extracellular')
#		sec.xraxial[0]=(rhoa*.01)/(np.pi*((((sec.diam/2)+space_p)**2)-((sec.diam/2)**2)))
#		sec.xraxial[1]=(rhoa*.01)/(np.pi*((((sec.diam/2)+space_p)**2)-((sec.diam/2)**2)))
#		sec.xg[0]=1e10 
#		sec.xg[1]=1e10 
#		sec.xc[0]=0
#		sec.xc[1]=0
		return(secref)


	
	def assign_hillock_biophysics(self,sec,rhoa,space_p,sectionlen,section_diam):
		sec.push()
		secref = h.SectionRef()
		if sectionlen is not None:
			sec.L = sectionlen
		if section_diam is not None:
			sec.diam = section_diam
		sec.Ra = self.PARAM_Ra
		sec.cm = self.PARAM_cm
		sec.nseg = self.dlambda(sec)
		sec.insert('pas')
		sec.g_pas=self.PARAM_g_pas
		sec.e_pas=-1*self.PARAM_e_pas
		sec.insert('SKv3_1')
		sec.insert('K_Pst')
		sec.insert('SK_E2')
		sec.insert('CaDynamics_E2')
		sec.insert('Im')
		sec.insert('Ih')
		sec.decay_CaDynamics_E2 = 460.0
		sec.gamma_CaDynamics_E2 = 0.000501
		sec.gSKv3_1bar_SKv3_1 = self.PARAM_gSKv3_1bar
		sec.gK_Pstbar_K_Pst = self.PARAM_gK_Pstbar
		sec.gSK_E2bar_SK_E2 = self.PARAM_gSK_E2bar
		sec.gImbar_Im = self.PARAM_gImbar
		sec.gIhbar_Ih = self.PARAM_gIhbar
#		sec.insert('extracellular')
#		sec.xraxial[0]=(rhoa*.01)/(np.pi*((((sec.diam/2)+space_p)**2)-((sec.diam/2)**2)))
#		sec.xraxial[1]=(rhoa*.01)/(np.pi*((((sec.diam/2)+space_p)**2)-((sec.diam/2)**2)))
#		sec.xg[0]=1e10 
#		sec.xg[1]=1e10 
#		sec.xc[0]=0
#		sec.xc[1]=0

		return(secref)
	
	def assign_initial_segment_biophysics(self,sec,rhoa,space_p,sectionlen,section_diam):
		sec.push()
		secref = h.SectionRef()
		sec.insert('pas')
		if sectionlen is not None:
			sec.L = sectionlen
		if section_diam is not None:
			sec.diam = section_diam
		sec.Ra = self.PARAM_Ra
		sec.cm = self.PARAM_cm
		sec.nseg = self.dlambda(sec)
		sec.insert('pas')
		sec.g_pas=self.PARAM_g_pas
		sec.e_pas=-1*self.PARAM_e_pas
		sec.insert('SKv3_1')
		sec.insert('K_Pst')
		sec.insert('SK_E2')
		sec.insert('CaDynamics_E2')
		sec.insert('Im')
		sec.insert('Ih')
		sec.decay_CaDynamics_E2 = 460.0
		sec.gamma_CaDynamics_E2 = 0.000501
		sec.gSKv3_1bar_SKv3_1 = self.PARAM_gSKv3_1bar
		sec.gK_Pstbar_K_Pst = self.PARAM_gK_Pstbar
		sec.gSK_E2bar_SK_E2 = self.PARAM_gSK_E2bar
		sec.gImbar_Im = self.PARAM_gImbar
		sec.gIhbar_Ih = self.PARAM_gIhbar
#		sec.insert('extracellular')
#		sec.xraxial[0]=(rhoa*.01)/(np.pi*((((sec.diam/2)+space_p)**2)-((sec.diam/2)**2)))
#		sec.xraxial[1]=(rhoa*.01)/(np.pi*((((sec.diam/2)+space_p)**2)-((sec.diam/2)**2)))
#		sec.xg[0]=1e10 
#		sec.xg[1]=1e10 
#		sec.xc[0]=0
#		sec.xc[1]=0

		return(secref)
	
	def assign_basal_biophysics(self,sec,rhoa,space_p,sectionlen,section_diam):
		sec.push()
		secref = h.SectionRef()
		if sectionlen is not None:
			sec.L = sectionlen
		if section_diam is not None:
			sec.diam = section_diam
		sec.Ra = self.PARAM_Ra
		sec.cm = self.PARAM_cm
		sec.nseg = self.dlambda(sec)
		sec.insert('pas')
		sec.g_pas=self.PARAM_g_pas
		sec.e_pas=-1*self.PARAM_e_pas
		sec.insert('Ih')
		sec.gIhbar_Ih = self.PARAM_gIhbar
#		sec.insert('extracellular')
#		sec.xraxial[0]=(rhoa*.01)/(np.pi*((((sec.diam/2)+space_p)**2)-((sec.diam/2)**2)))
#		sec.xraxial[1]=(rhoa*.01)/(np.pi*((((sec.diam/2)+space_p)**2)-((sec.diam/2)**2)))
#		sec.xg[0]=1e10 
#		sec.xg[1]=1e10 
#		sec.xc[0]=0
#		sec.xc[1]=0
		return(secref)
	
	def assign_apical_biophysics(self,sec,rhoa,space_p,sectionlen,section_diam):
		sec.push()
		dist_reset = h.distance()
		distance=h.distance(self.soma[0](0.5),sec=sec)
		secref = h.SectionRef()
		if sectionlen is not None:
			sec.L = sectionlen
		if section_diam is not None:
			sec.diam = section_diam
		sec.Ra = self.PARAM_Ra
		sec.cm = self.PARAM_cm
		sec.nseg = self.dlambda(sec)
		sec.insert('pas')
		sec.g_pas=self.PARAM_g_pas
		sec.e_pas=-1*self.PARAM_e_pas
		sec.insert('Ca_LVAst')
		sec.insert('Ca_HVA')
		sec.insert('SKv3_1')
		sec.insert('SK_E2')
		sec.insert('CaDynamics_E2')
		sec.insert('Ih')
		sec.insert('Im')
		sec.insert('NaTa_t')
		sec.decay_CaDynamics_E2 = 460.0
		sec.gamma_CaDynamics_E2 = 0.000501
		sec.gSKv3_1bar_SKv3_1 = self.PARAM_gSKv3_1bar
		sec.gSK_E2bar_SK_E2 = self.PARAM_gSK_E2bar
		sec.gImbar_Im = self.PARAM_gImbar
		sec.gNaTa_tbar_NaTa_t = self.PARAM_NaTa_t_apical
		dist = h.distance(0.5)
		sec.gIhbar_Ih = self.PARAM_gIhbar*(-0.8696+2.0870*np.exp(3.6161*(dist/self.PARAM_decay)))
		if dist>360 and dist < 600:
			sec.gCa_LVAstbar_Ca_LVAst = 100*self.PARAM_gCa_LVAstbar
			sec.gCa_HVAbar_Ca_HVA = 10*self.PARAM_gCa_HVAbar
		else:
			sec.gCa_LVAstbar_Ca_LVAst = self.PARAM_gCa_LVAstbar
			sec.gCa_HVAbar_Ca_HVA = self.PARAM_gCa_HVAbar
		
#		sec.insert('extracellular')
#		sec.xraxial[0]=(rhoa*.01)/(np.pi*((((sec.diam/2)+space_p)**2)-((sec.diam/2)**2)))
#		sec.xraxial[1]=(rhoa*.01)/(np.pi*((((sec.diam/2)+space_p)**2)-((sec.diam/2)**2)))
#		sec.xg[0]=1e10 
#		sec.xg[1]=1e10 
#		sec.xc[0]=0
#		sec.xc[1]=0
		return(secref)

	def assign_tuft_biophysics(self,sec,rhoa,space_p,sectionlen,section_diam):
		sec.push()
		dist_reset = h.distance()
		distance=h.distance(self.soma[0](0.5),sec=sec)
		secref = h.SectionRef()
		if sectionlen is not None:
			sec.L = sectionlen
		if section_diam is not None:
			sec.diam = section_diam
		sec.Ra = self.PARAM_Ra
		sec.cm = self.PARAM_cm
		sec.nseg = self.dlambda(sec)
		sec.insert('pas')
		sec.g_pas=self.PARAM_g_pas
		sec.e_pas=-1*self.PARAM_e_pas
		sec.insert('Ca_LVAst')
		sec.insert('Ca_HVA')
		sec.insert('SKv3_1')
		sec.insert('SK_E2')
		sec.insert('CaDynamics_E2')
		sec.insert('Ih')
		sec.insert('Im')
		sec.insert('NaTa_t')
		sec.decay_CaDynamics_E2 = 460.0
		sec.gamma_CaDynamics_E2 = 0.000501
		sec.gSKv3_1bar_SKv3_1 = self.PARAM_gSKv3_1bar
		sec.gSK_E2bar_SK_E2 = self.PARAM_gSK_E2bar
		sec.gImbar_Im = self.PARAM_gImbar
		sec.gNaTa_tbar_NaTa_t = self.PARAM_NaTa_t_apical
		dist = h.distance(0.5)
		sec.gIhbar_Ih = self.PARAM_gIhbar*(-0.8696+2.0870*np.exp(3.6161*(dist/self.PARAM_decay)))
		if dist>360 and dist < 600:
			sec.gCa_LVAstbar_Ca_LVAst = 100*self.PARAM_gCa_LVAstbar
			sec.gCa_HVAbar_Ca_HVA = 10*self.PARAM_gCa_HVAbar
		else:
			sec.gCa_LVAstbar_Ca_LVAst = self.PARAM_gCa_LVAstbar
			sec.gCa_HVAbar_Ca_HVA = self.PARAM_gCa_HVAbar
		
#		sec.insert('extracellular')
#		sec.xraxial[0]=(rhoa*.01)/(np.pi*((((sec.diam/2)+space_p)**2)-((sec.diam/2)**2)))
#		sec.xraxial[1]=(rhoa*.01)/(np.pi*((((sec.diam/2)+space_p)**2)-((sec.diam/2)**2)))
#		sec.xg[0]=1e10 
#		sec.xg[1]=1e10 
#		sec.xc[0]=0
#		sec.xc[1]=0
		return(secref)
	
	def assign_bouton_biophysics(self,sec,rhoa,space_p,sectionlen):
		sec.push()
		secref = h.SectionRef()
		sec.L = sectionlen
		sec.nseg = 1
		sec.diam = 2
		sec.Ra = rhoa/10000
		sec.cm = 1.0
		sec.insert('na12')
		sec.gbar_na12 = 1000.0
		sec.insert('axnodeX')
		sec.gnapbar_axnodeX = 0.0024378560807969537
		sec.gnabar_axnodeX = 4.24219598883859
		sec.gkbar_axnodeX = 0.12977029013875638
		sec.gl_axnodeX = 0.007
		sec.ena_axnodeX = 50.0
		sec.ek_axnodeX = -90.0
		sec.el_axnodeX = -90.0
#		sec.insert('extracellular')
#		sec.xraxial[0]=(rhoa*.01)/(np.pi*((((sec.diam/2)+space_p)**2)-((sec.diam/2)**2)))
#		sec.xraxial[1]=(rhoa*.01)/(np.pi*((((sec.diam/2)+space_p)**2)-((sec.diam/2)**2)))
#		sec.xg[0]=1e10 
#		sec.xg[1]=1e10 
#		sec.xc[0]=0
#		sec.xc[1]=0
		return(secref)
	
	def assign_interbouton_biophysics(self,sec,rhoa,space_p,sectionlen):
		sec.push()
		secref = h.SectionRef()
		sec.L = sectionlen
		sec.nseg = 1
		sec.diam = 2
		sec.Ra = rhoa/10000
		sec.cm = 1.0
		sec.insert('na12')
		sec.gbar_na12 = 1000.0
		sec.insert('axnodeX')
		sec.gnapbar_axnodeX = 0.00370109160245544
		sec.gnabar_axnodeX = 3.068875056920868
		sec.gkbar_axnodeX = 0.10527073936782891
		sec.gl_axnodeX = 0.007
		sec.ena_axnodeX = 50.0
		sec.ek_axnodeX = -90.0
		sec.el_axnodeX = -90.0
#		sec.insert('extracellular')
#		sec.xraxial[0]=(rhoa*.01)/(np.pi*((((sec.diam/2)+space_p)**2)-((sec.diam/2)**2)))
#		sec.xraxial[1]=(rhoa*.01)/(np.pi*((((sec.diam/2)+space_p)**2)-((sec.diam/2)**2)))
#		sec.xg[0]=1e10 
#		sec.xg[1]=1e10 
#		sec.xc[0]=0
#		sec.xc[1]=0
		return(secref)
	
	def assign_nodal_biophysics(self,sec,rhoa,space_p,outerdiam,innerdiam,sectionlen):
		sec.push()
		secref = h.SectionRef()
		sec.L = sectionlen
		sec.nseg = 1
		sec.diam = innerdiam
		sec.Ra = rhoa/10000
		sec.cm = 2.0
		sec.insert('axnode')
		sec.gnapbar_axnode = 0.005
		sec.gnabar_axnode = 3.0
		sec.gkbar_axnode = 0.08
		sec.gl_axnode = 0.007
		sec.ena_axnode = 50.0
		sec.ek_axnode = -90.0
		sec.el_axnode = -90.0
#		sec.insert('extracellular')
#		sec.xraxial[0]=(rhoa*.01)/(np.pi*((((innerdiam/2)+space_p)**2)-((innerdiam/2)**2)))
#		sec.xraxial[1]=(rhoa*.01)/(np.pi*((((innerdiam/2)+space_p)**2)-((innerdiam/2)**2)))
#		sec.xg[0]=1e10 
#		sec.xg[1]=1e10 
#		sec.xc[0]=0
#		sec.xc[1]=0
		return(secref)
	
	def assign_paranode1_biophysics(self,sec,rhoa,space_p,outerdiam,innerdiam,nl,sectionlen):
		sec.push()
		secref = h.SectionRef()
		sec.L = sectionlen
		sec.nseg = 1
		sec.diam = outerdiam
		sec.Ra = rhoa/10000
		sec.cm = 2*innerdiam/outerdiam
		sec.insert('pas')
		sec.g_pas = 0.001*innerdiam/outerdiam
		sec.e_pas = h.v_init
#		sec.insert('extracellular')
#		sec.xraxial[0]=(rhoa*.01)/(np.pi*((((innerdiam/2)+space_p)**2)-((innerdiam/2)**2)))
#		sec.xraxial[1]=(rhoa*.01)/(np.pi*((((innerdiam/2)+space_p)**2)-((innerdiam/2)**2)))
#		sec.xg[0]=0.001/(nl*2)
#		sec.xg[1]=0.001/(nl*2)
#		sec.xc[0]=0.1/(nl*2)
#		sec.xc[1]=0.1/(nl*2)
		return(secref)
	
	def assign_paranode2_biophysics(self,sec,rhoa,space_p,outerdiam,innerdiam,nl,sectionlen):
		sec.push()
		secref = h.SectionRef()
		sec.L = sectionlen
		sec.nseg = 1
		sec.diam = outerdiam
		sec.Ra = rhoa/10000
		sec.cm = 2.0*innerdiam/outerdiam
		sec.insert('pas')
		sec.g_pas = 0.0001*innerdiam/outerdiam
		sec.e_pas = h.v_init
#		sec.insert('extracellular')
#		sec.xraxial[0]=(rhoa*.01)/(np.pi*((((innerdiam/2)+space_p)**2)-((innerdiam/2)**2)))
#		sec.xraxial[1]=(rhoa*.01)/(np.pi*((((innerdiam/2)+space_p)**2)-((innerdiam/2)**2)))
#		sec.xg[0]=0.001/(nl*2)
#		sec.xg[1]=0.001/(nl*2)
#		sec.xc[0]=0.1/(nl*2)
#		sec.xc[1]=0.1/(nl*2)
		return(secref)
	
	def assign_internode_biophysics(self,sec,rhoa,space_p,outerdiam,innerdiam,nl,sectionlen):
		sec.push()
		secref = h.SectionRef()
		sec.L = sectionlen
		sec.nseg = 3
		sec.diam = outerdiam
		sec.Ra = rhoa/10000
		sec.cm = 2.0*innerdiam/outerdiam
		sec.insert('pas')
		sec.g_pas = 0.0001*innerdiam/outerdiam
		sec.e_pas = h.v_init
#		sec.insert('extracellular')
#		sec.xraxial[0]=(rhoa*.01)/(np.pi*((((innerdiam/2)+space_p)**2)-((innerdiam/2)**2)))
#		sec.xraxial[1]=(rhoa*.01)/(np.pi*((((innerdiam/2)+space_p)**2)-((innerdiam/2)**2)))
#		sec.xg[0]=0.001/(nl*2)
#		sec.xg[1]=0.001/(nl*2)
#		sec.xc[0]=0.1/(nl*2)
#		sec.xc[1]=0.1/(nl*2)
		return(secref)
	
	def modify_somatodendritic_biophysics(self,bphys):
		h.shift_NaTa_t = bphys['NaTaShift'] #--> h.shift_NaTa_t --> 5
		h.shift_Nap_Et2 = bphys['NapShift'] #--> h.shift_Nap_Et2 --> 0
		h.shift_K_Pst = bphys['Kshift'] #--> h.shift_K_Pst=0
		h.shift_SKv3_1 = bphys['SKv3Shift'] #--> h.shift_SKv3_1 --> 10
		h.a_Ih = bphys['aIh'] #--> h.a_Ih --> 23.428
		h.b_Ih= bphys['bIh'] #--> h.b_Ih --> 0.21756
		h.c_Ih = bphys['cIh'] #--> h.c_Ih --> 1.3881e-09
		h.d_Ih = bphys['dIh'] #--> h.d_Ih --> 0.082329
		h.e_Ih = bphys['eIh'] #--> h.e_Ih --> 1.9419e-09
		h.k_Ih = bphys['kIh'] #--> h.k_Ih --> 8.0775
		h.vh_Ih = bphys['vhIh'] #--> h.vh_Ih --> -90.963
		h.ehcn_Ih = bphys['ehcnIh'] #--> h.ehcn_Ih --> -49.765
		for sec in self.hillock:
			sec.Ra = bphys['ra_hillock']
			sec.cm = bphys['cm_hillock']
			sec.g_pas=bphys['gpas_hillock']
			sec.e_pas=-1*bphys['epas_hillock']
			sec.decay_CaDynamics_E2 = bphys['CaDynamics']
			sec.gamma_CaDynamics_E2 = bphys['gamma']
			sec.gSKv3_1bar_SKv3_1 = bphys['gSKv3']
			sec.gK_Pstbar_K_Pst = bphys['gK'] #self.PARAM_gK_Pstbar
			sec.gSK_E2bar_SK_E2 = bphys['gSK'] #self.PARAM_gSK_E2bar
			sec.gImbar_Im = bphys['gIm'] #self.PARAM_gImbar
			sec.gIhbar_Ih = bphys['gIh'] #self.PARAM_gIhbar
		for sec in self.soma:
			sec.Ra = bphys['ra_soma']#self.PARAM_Ra
			sec.cm = bphys['cm_soma']#self.axosomatic_cm 
			sec.g_pas=bphys['gpas_soma']#self.PARAM_g_pas
			sec.e_pas=-1*bphys['epas_soma']#self.PARAM_e_pas
			sec.gCa_LVAstbar_Ca_LVAst = bphys['gCaLVA']#self.PARAM_gCa_LVAstbar
			sec.gCa_HVAbar_Ca_HVA = bphys['gCaHVA']#self.PARAM_gCa_HVAbar
			sec.decay_CaDynamics_E2 = bphys['CaDynamics']#460.0
			sec.gamma_CaDynamics_E2 = bphys['gamma']#0.000501
			sec.gSKv3_1bar_SKv3_1 = bphys['gSKv3']#self.PARAM_gSKv3_1bar
			sec.gK_Pstbar_K_Pst = bphys['gK']#self.PARAM_gK_Pstbar
			sec.gK_Tstbar_K_Tst = bphys['gKTst']#self.PARAM_gK_Tstbar
			sec.gSK_E2bar_SK_E2 = bphys['gSK']#self.PARAM_gSK_E2bar
			sec.gIhbar_Ih = bphys['gIh']#self.PARAM_gIhbar
			sec.gNap_Et2bar_Nap_Et2 = bphys['Nap']#self.PARAM_Nap_Et2
			sec.gNaTa_tbar_NaTa_t = bphys['NaTasoma']#self.PARAM_NaTa_t_soma
		for sec in self.basal:
			sec.Ra = bphys['ra_basal']#self.PARAM_Ra
			sec.cm = bphys['cm_basal']#self.PARAM_cm
			sec.g_pas=bphys['gpas_basal']#self.PARAM_g_pas
			sec.e_pas=-1*bphys['epas_basal']#self.PARAM_e_pas
			sec.gIhbar_Ih = bphys['gIh']#self.PARAM_gIhbar
		for sec in self.apical:
			sec.Ra = bphys['ra_apical']#self.PARAM_Ra
			sec.cm = bphys['cm_apical']#self.PARAM_cm
			sec.g_pas=bphys['gpas_apical']#self.PARAM_g_pas
			sec.e_pas=-1*bphys['epas_apical']#self.PARAM_e_pas
			sec.decay_CaDynamics_E2 = bphys['CaDynamics']#460.0
			sec.gamma_CaDynamics_E2 = bphys['gamma']#0.000501
			sec.gSKv3_1bar_SKv3_1 = bphys['gSKv3']#self.PARAM_gSKv3_1bar
			sec.gSK_E2bar_SK_E2 = bphys['gSK']#self.PARAM_gSK_E2bar
			sec.gImbar_Im = bphys['gIm']#self.PARAM_gImbar
			sec.gNaTa_tbar_NaTa_t = bphys['NaTaapical']#self.PARAM_NaTa_t_apical
			dist = h.distance(0.5)
			sec.gIhbar_Ih = bphys['gIh']*(-0.8696+2.0870*np.exp(3.6161*(dist/bphys['decay']))) #self.PARAM_gIhbar*(-0.8696+2.0870*np.exp(3.6161*(dist/self.PARAM_decay)))
			if dist>360 and dist < 600:
				sec.gCa_LVAstbar_Ca_LVAst = 100*bphys['gCaLVA']#self.PARAM_gCa_LVAstbar
				sec.gCa_HVAbar_Ca_HVA = 10*bphys['gCaHVA']#self.PARAM_gCa_HVAbar
			else:
				sec.gCa_LVAstbar_Ca_LVAst = 100*bphys['gCaLVA']#self.PARAM_gCa_LVAstbar
				sec.gCa_HVAbar_Ca_HVA = bphys['gCaHVA']#self.PARAM_gCa_HVAbar
		for sec in self.tuft:
			pass
#			sec.Ra = bphys['ra_apical']#self.PARAM_Ra
#			sec.cm = bphys['ra_apical']#self.PARAM_cm
#			sec.g_pas=bphys['gpas_apical']#self.PARAM_g_pas
#			sec.e_pas=-1*bphys['epas_apical']#self.PARAM_e_pas
#			sec.decay_CaDynamics_E2 = bphys['CaDynamics']#460.0
#			sec.gamma_CaDynamics_E2 = bphys['gamma']#0.000501
#			sec.gSKv3_1bar_SKv3_1 = bphys['gSKv3']#self.PARAM_gSKv3_1bar
#			sec.gSK_E2bar_SK_E2 = bphys['gSK']#self.PARAM_gSK_E2bar
#			sec.gImbar_Im = bphys['gIm']#self.PARAM_gImbar
#			sec.gNaTa_tbar_NaTa_t = bphys['NaTaapical']#self.PARAM_NaTa_t_apical
#			dist = h.distance(0.5)
#			sec.gIhbar_Ih = bphys['gIh']*(-0.8696+2.0870*np.exp(3.6161*(dist/bphys['decay']))) #self.PARAM_gIhbar*(-0.8696+2.0870*np.exp(3.6161*(dist/self.PARAM_decay)))
#			if dist>360 and dist < 600:
#				sec.gCa_LVAstbar_Ca_LVAst = 100*bphys['gCaLVA']#self.PARAM_gCa_LVAstbar
#				sec.gCa_HVAbar_Ca_HVA = 10*bphys['gCaHVA']#self.PARAM_gCa_HVAbar
#			else:
#				sec.gCa_LVAstbar_Ca_LVAst = bphys['gCaLVA']#self.PARAM_gCa_LVAstbar
#				sec.gCa_HVAbar_Ca_HVA = bphys['gCaHVA']#self.PARAM_gCa_HVAbar
		return()
	
	def modify_axon_biophysics(self,bphys):
		for section in self.interbouton:
			section.gkbar_axnodeX = bphys['k_ib']
			section.gnabar_axnodeX = bphys['na_ib']
			section.gnapbar_axnodeX = 0.0
			section.gl_axnodeX = bphys['gl_ib']
			section.gbar_na12 = bphys['gbar_ib']
		
		for section in self.bouton:
			section.gkbar_axnodeX = bphys['k_b']
			section.gnabar_axnodeX = bphys['na_b']
			section.gnapbar_axnodeX = 0.0
			section.gl_axnodeX = bphys['gl_b']
			section.gbar_na12 = bphys['gbar_b']
		
		for section in self.node:
			section.gkbar_axnode = bphys['k_nb']
			section.gnabar_axnode = bphys['na_nb']
			section.gnapbar_axnode = bphys['gnap_nb']
			section.gl_axnode = bphys['gl_nb']
		
		h.amA_axnodeX = bphys['amA']
		h.amB_axnodeX = bphys['amB']
		h.amC_axnodeX = bphys['amC']
		h.bmA_axnodeX = bphys['bmA']
		h.bmB_axnodeX = bphys['bmB']
		h.bmC_axnodeX = bphys['bmC']
		h.ahA_axnodeX = bphys['ahA']
		h.ahB_axnodeX = bphys['ahB']
		h.ahC_axnodeX = bphys['ahC']
		h.asA_axnodeX = bphys['asA']
		h.asB_axnodeX = bphys['asB']
		h.asC_axnodeX = bphys['asC']
		
		h.amA_axnode = bphys['amA_n']
		h.amB_axnode = bphys['amB_n']
		h.amC_axnode = bphys['amC_n']
		h.bmA_axnode = bphys['bmA_n']
		h.bmB_axnode = bphys['bmB_n']
		h.bmC_axnode = bphys['bmC_n']
		h.ahA_axnode = bphys['ahA_n']
		h.ahB_axnode = bphys['ahB_n']
		h.ahC_axnode = bphys['ahC_n']
		h.asA_axnode = bphys['asA_n']
		h.asB_axnode = bphys['asB_n']
		h.asC_axnode = bphys['asC_n']
		return()
