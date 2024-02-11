import matplotlib.pyplot as plt
import numpy as np
import pickle
import multiprocessing
from multiprocessing.pool import ThreadPool
import subprocess
import os
import time

class Evolve():

	'''
	Evolve is a an evolutionary algorithm which is designed be be independent from model execution and evaluation to provide maximum flexibility in cost function design. 
	
	Evolve interacts with models/cost functions through the class method RunCellScript(), which must accept command line arguments which correspond to parameters (global or local) which are used to update the model prior to execution. The model/cost function script name is currently hard coded within this class method but should correspond to wherever you built your model and evaluate its performance. Your model file/cost function script should output parameters used and model score to a file named 'parallelresults.txt' using a threadsafe append type of function so that ENBO can safely read them.
	
	
	Evolve expects the following parameters (required):
	
	modelscript: str - pathname of file which runs and evaluates your model
	param_ranges: list - [[min,max,increment],[min,max,increment],...]
	
	But also accepts the following parameters (optional):
	
	threshold_type: str, 'pool','individual' - this parameter designates whether it is pool or individual performances which triggers ENBO termination criteria
	score_threshold: int/float - this parameter should correspond to your desired cost function performance.
	generation_size: int - the number of individuals to be evaluated in parallel in each generation. Should not be greater than the number of processors
	mating_pool_size: int - the size of your mating pool should be adjusted according to the size/complexity of your open parameter space
	generation_limit: int - the number of generations (*generation_size) that will be evaluated before ENBO timeout
	dynamic_mutation: boolean - should mutation rate be ajusted according to mean pool score - mutation rate determined by class method update_mutation_rate() and ranges from 0.5-0.0
	mutation_rate: float - initial mutation rate. Must be a value between 1.0-0.0
	seed_generation: li - List of list of parameters equal in length to mating_pool_size which may be used to initialize Evolve.
	
	Evolve creates and manages two datafiles:
	
	'MatingPool.pickle' - stores the current mating pool
	'best_individual.txt' - following each generation ENBO stores the parameters for the best performing individual in the pool and the mean pool score
	'''
	
	def __init__(self,modelscript=None, param_ranges = [],outputdir=os.getcwd(),threshold_type='pool',score_threshold=1,generation_size=19,mating_pool_size=190,generation_limit=10,dynamic_mutation=True,mutation_rate=0.5,seed_generation=None):
		start_time = time.time()
		self.modelscript = modelscript
		self.outputdir = outputdir
		if self.modelscript is None:
			print('No modelscript provided')
			return()
		
		self.threshold_type = threshold_type
		self.threshold=False
		self.max_pool_size = mating_pool_size
		self.score_threshold=score_threshold
		self.check_set_generation_size(generation_size)
		self.generation_limit = generation_limit
		self.generations = 0
		self.param_ranges = param_ranges
		if self.param_ranges == []:
			return()
		
		self.initialize_mating_pool(seed_generation=seed_generation)
		print('Pool initialized')
		end_time = time.time()
		self.time_it(start_time)
		self.original_pool_mean = np.mean(self.mating_pool_evaluations)
		print('Original pool mean score: ',self.original_pool_mean)
		self.dynamic_mutation = dynamic_mutation
		if self.dynamic_mutation:
			self.mutation_rate = 0.5
		else:
			self.mutation_rate = mutation_rate	
		
		print('Original mutation rate: ',self.mutation_rate)	
		
	def check_set_generation_size(self,gen_size):
		if gen_size > multiprocessing.cpu_count():
			gen_size = multiprocessing.cpu_count()
			print('It is not recommended to have generation size greater than number of processors on machine. Reduced generation size to cpu count -1.')
		
		self.generation_size = gen_size
	
	def initialize_mating_pool(self,seed_generation=None):
		if seed_generation is not None:
			self.mating_pool = seed_generation
		else:
			self.mating_pool = []
			print('Initialized mating pool with '+str(self.max_pool_size)+' random entries')
		
		while len(self.mating_pool) < self.max_pool_size:
			self.mating_pool.append(self.get_random_individual())
		
		
		self.mating_pool_evaluations = []
		for group in self.divide_chunks(self.mating_pool,self.generation_size):
			self.mating_pool_evaluations+=self.evaluate_generation(group)
		
	
	def sort_children_by_varbs(self,children,varbs,spikes):
		newspikes = []
		newvarbs = []
		for child in children:
			which = varbs.index(child)
			newvarbs.append(varbs[which])
			newspikes.append(spikes[which])
		
		return(newvarbs,newspikes)
	
	def update_threshold(self,scores):
		if self.threshold_type == 'pool' and self.mating_pool_evaluations != []:
			if np.mean(self.mating_pool_evaluations) < self.score_threshold:
				self.threshold=True
		
		else:
			for score in scores:
				if score[0] < self.threshold:
					self.threshold = True
	
	def evaluate_generation(self,children):
		open('parallelresults.txt','w').close()
		pool = ThreadPool(self.generation_size)
		pool.map(self.RunCellScript,children)
		varbs,scores = self.load_parallel_results()
		varbs,scores = self.sort_children_by_varbs(children,varbs,scores)
		print(scores)
		self.update_threshold(scores)
		return([score[0] for score in scores])
	
	def divide_chunks(self, l, n):
		for i in range(0,len(l),n):
			yield l[i:i+n]
	
	def check_add_to_mating_pool(self,children,scores):
		for c,child in enumerate(children):
			pool_max = np.max(self.mating_pool_evaluations)
			if scores[c] < pool_max:
				individual_to_remove = self.mating_pool_evaluations.index(pool_max)
				self.mating_pool_evaluations[individual_to_remove] = scores[c]
				self.mating_pool[individual_to_remove] = child

	def RunCellScript(self,params):
		args = 'python3 '+self.modelscript+' '
		for p in params:
			args+=str(p)+' '
		
		subprocess.call(args,shell=True)
	
	def parse_results_line(self,line):
		return([float(item) for item in line.strip('[]\n').split(',')])

	def load_parallel_results(self):
		with open('parallelresults.txt','r') as f:
			results = [self.parse_results_line(item) for item in f.readlines()]
		
		varbs = results[::2]
		sptimes = results[1::2]
		return(varbs,sptimes)
	
	def mating_pool_snapshot(self):
		pool_min = np.min(self.mating_pool_evaluations)
		best_individual_index = self.mating_pool_evaluations.index(pool_min)
		best_score = self.mating_pool_evaluations[best_individual_index]
		best_individual = self.mating_pool[best_individual_index]
		try:
			with open(self.outputdir+'best_individual.txt','a') as f:
				f.write(str(best_score)+'\n')
				f.write(str(best_individual)+'\n')		
				f.write(str(np.mean(self.mating_pool_evaluations))+'\n')
				print(best_individual,'best_individual',best_score,'best_score')
		
		except:
			with open(self.outputdir+'best_individual.txt','w') as f:
				f.write(str(best_score)+'\n')
				f.write(str(best_individual)+'\n')
				f.write(str(np.mean(self.mating_pool_evaluations))+'\n')
		
		with open(self.outputdir+'MatingPool.pickle','wb') as f:
			pickle.dump([self.mating_pool,self.mating_pool_evaluations],f)
	
	def get_random_individual(self):
		A = []
		for pr in self.param_ranges:
			A.append(np.random.choice(np.arange(pr[0],pr[1],pr[2])))
		
		return(A)
	
	def get_new_generation(self):
		children = []
		while len(children) < self.generation_size:
			mom,dad = [self.mating_pool[i] for i in np.random.choice(range(self.max_pool_size),2)]
			children+=self.crossover(mom,dad)
		
		return([self.mutate(child) for child in children[:self.generation_size]])
	
	def crossover(self,varsA,varsB):
		crossover_point = np.random.choice(range(len(varsA)))
		if np.random.random() > 0.5:
			newA = varsA[:crossover_point]+varsB[crossover_point:]
			newB = varsB[:crossover_point]+varsA[crossover_point:]
		
		else:
			newB = varsA[:crossover_point]+varsB[crossover_point:]
			newA = varsB[:crossover_point]+varsA[crossover_point:]
		
		newC = [(varsA[i]+varsB[i])/2.0 for i in range(len(varsA))]
		return(newA,newB,newC)
	
	def mutate(self,varbs):
		for v,var in enumerate(varbs):
			if np.random.random() < self.mutation_rate:
				varbs[v]=np.random.uniform(self.param_ranges[v][0],self.param_ranges[v][1])
				if varbs[v] < self.param_ranges[v][0]:
					varbs[v] = self.param_ranges[v][0]
				
				if varbs[v] > self.param_ranges[v][1]:
					varbs[v] = self.param_ranges[v][1]
		
		return(varbs)
	
	def update_mutation_rate(self):
		pool_mean = np.mean(self.mating_pool_evaluations)
		print('New pool mean score: ',pool_mean)
		self.mutation_rate = 0.5/(1+np.exp(0.5+(-1*(pool_mean*100))/25))		#0.5/(1+np.exp(1+(-1*pool_mean)/25))
		print('New mutation rate: ',self.mutation_rate)
	
	def time_it(self,s):
		e = time.time()
		print('elapsed time: '+str(round((e-s)/60.0,1))+' minutes')
	
	def run(self):
		start_time = time.time()
		self.threshold=False
		while not self.threshold and self.generations < self.generation_limit:
			self.mating_pool_snapshot()
			children = self.get_new_generation()
			scores = self.evaluate_generation(children)
			self.check_add_to_mating_pool(children,scores)
			if self.dynamic_mutation:
				self.update_mutation_rate()
			
			self.time_it(start_time)
			self.generations+=1
		
		self.mating_pool_snapshot()


if __name__ == "__main__":
	params = [
		[350,650,3], #bphys['ra_soma'] --> self.PARAM_Ra --> 501.6
		[350,650,3], #bphys['ra_basal'] --> self.PARAM_Ra --> 501.6
		[350,650,3], #bphys['ra_apical'] --> self.PARAM_Ra --> 501.6
		[350,650,3], #bphys['ra_hillock'] --> self.PARAM_Ra --> 501.6
		[1.6226/2.0,1.6226*2.0,np.abs((1.6226*2.0-1.6226/2.0))/100.0], #bphys['cm_soma'] --> self.PARAM_cm --> 1.6226
		[1.6226/2.0,1.6226*2.0,np.abs((1.6226*2.0-1.6226/2.0))/100.0], #bphys['cm_basal'] --> self.PARAM_cm --> 1.6226
		[1.6226/2.0,1.6226*2.0,np.abs((1.6226*2.0-1.6226/2.0))/100.0], #bphys['cm_apical'] --> self.PARAM_cm --> 1.6226
		[1.6226/2.0,1.6226*2.0,np.abs((1.6226*2.0-1.6226/2.0))/100.0], #bphys['cm_hillock'] --> self.PARAM_cm --> 1.6226
		[1.75e-05/2.0,1.75e-05*2.0,np.abs((1.75e-05*2.0-1.75e-05/2.0))/100.0], #bphys['gpas_soma'] --> self.PARAM_g_pas --> 1.75e-05
		[1.75e-05/2.0,1.75e-05*2.0,np.abs((1.75e-05*2.0-1.75e-05/2.0))/100.0], #bphys['gpas_basal'] --> self.PARAM_g_pas --> 1.75e-05
		[1.75e-05/2.0,1.75e-05*2.0,np.abs((1.75e-05*2.0-1.75e-05/2.0))/100.0], #bphys['gpas_apical'] --> self.PARAM_g_pas --> 1.75e-05
		[1.75e-05/2.0,1.75e-05*2.0,np.abs((1.75e-05*2.0-1.75e-05/2.0))/100.0], #bphys['gpas_hillock'] --> self.PARAM_g_pas --> 1.75e-05
		[460.0/2.0,460.0*2.0,np.abs((460.0*2.0-460.0/2.0))/100.0], #bphys['CaDynamics'] --> hillock_CaDynamics_E2 --> 460.0
		[0.000501/2.0,0.000501*2.0,np.abs((0.000501*2.0-0.000501/2.0))/100.0], #bphys['gamma'] --> hillock_gamma_CaDynamics_E2 --> 0.000501
		[0.04/2.0,0.04*2.0,np.abs((0.04*2.0-0.04/2.0))/100.0], #bphys['gSKv3'] --> self.PARAM_gSKv3_1bar --> 0.04
		[0.07/2.0,0.07*2.0,np.abs((0.07*2.0-0.07/2.0))/100.0], #bphys['gK'] --> self.PARAM_gK_Pstbar --> 0.07
		[2.0964e-09/2.0,2.0964e-09*2.0,np.abs((2.0964e-09*2.0-2.0964e-09/2.0))/100.0], #bphys['gSK'] --> self.PARAM_gSK_E2bar --> 2.0964e-09
		[.0002/2.0,.0002*2.0,np.abs((.0002*2.0-.0002/2.0))/100.0], #bphys['gIm'] --> self.PARAM_gImbar --> .0002
		[5.0723e-05/2.0,5.0723e-05*2.0,np.abs((5.0723e-05*2.0-5.0723e-05/2.0))/100.0], #bphys['gIh'] --> self.PARAM_gIhbar --> 5.0723e-05
		[84.325/2.0,84.325*2.0,np.abs((84.325*2.0-84.325/2.0))/100.0], #bphys['epas_soma'] --> self.PARAM_e_pas --> 84.325
		[84.325/2.0,84.325*2.0,np.abs((84.325*2.0-84.325/2.0))/100.0], #bphys['epas_basal'] --> self.PARAM_e_pas --> 84.325
		[84.325/2.0,84.325*2.0,np.abs((84.325*2.0-84.325/2.0))/100.0], #bphys['epas_apical'] --> self.PARAM_e_pas --> 84.325
		[84.325/2.0,84.325*2.0,np.abs((84.325*2.0-84.325/2.0))/100.0], #bphys['epas_hillock'] --> self.PARAM_e_pas --> 84.325
		[1.5/2.0,1.5*2.0,np.abs((1.5*2.0-1.5/2.0))/100.0], #bphys['ascm'] --> self.axosomatic_cm --> 1.5
		[0.00099587/2.0,0.00099587*2.0,np.abs((0.00099587*2.0-0.00099587/2.0))/100.0], #bphys['gCaLVA'] --> self.PARAM_gCa_LVAstbar --> 0.00099587
		[1.7838e-09/2.0,1.7838e-09*2.0,np.abs((1.7838e-09*2.0-1.7838e-09/2.0))/100.0], #bphys['gCaHVA'] --> self.PARAM_gCa_HVAbar --> 1.7838e-09
		[2e-05/2.0,2e-05*2.0,np.abs((2e-05*2.0-2e-05/2.0))/100.0], #bphys['gKTst'] --> self.PARAM_gK_Tstbar --> 2e-05
		[1e-06/2.0,1e-06*2.0,np.abs((1e-06*2.0-1e-06/2.0))/100.0], #bphys['Nap'] --> self.PARAM_Nap_Et2 --> 1e-06
		[2.2/2.0,2.2*2.0,np.abs((2.2*2.0-2.2/2.0))/100.0], #bphys['NaTasoma'] --> self.PARAM_NaTa_t_soma --> 2.2
		[0.001/2.0,0.001*2.0,np.abs((0.001*2.0-0.001/2.0))/100.0], #bphys['NaTaapical'] --> self.PARAM_NaTa_t_apical --> 0.001
		[1000/2.0,1000*2.0,np.abs((1000*2.0-1000/2.0))/100.0], #bphys['decay'] --> self.PARAM_decay --> 1000
		[5/2.0, 5*2.0,np.abs(( 5*2.0 - 5/2.0))/100.0], #bphys['NaTaShift'] --> h.shift_NaTa_t --> 5
		[-1,1,2/100.0], #bphys['NapShift'] --> h.shift_Nap_Et2 --> 0
		[-1,1,2/100.0], #bphys['Kshift'] --> h.shift_K_Pst=0
		[0/2.0,10*2.0,np.abs((10*2.0-10/2.0))/100.0], #bphys['SKv3Shift'] --> h.shift_SKv3_1 --> 10
		[23.428/2.0,23.428*2.0,np.abs((23.428*2.0-23.428/2.0))/100.0], #bphys['aIh'] --> h.a_Ih --> 23.428
		[0.21756/2.0,0.21756*2.0,np.abs((0.21756*2.0-0.21756/2.0))/100.0], #bphys['bIh'] --> h.b_Ih --> 0.21756
		[1.3881e-09/2.0,1.3881e-09*2.0,np.abs((1.3881e-09*2.0-1.3881e-09/2.0))/100.0], #bphys['cIh'] --> h.c_Ih --> 1.3881e-09
		[0.082329/2.0,0.082329*2.0,np.abs((0.082329*2.0-0.082329/2.0))/100.0], #bphys['dIh'] --> h.d_Ih --> 0.082329
		[1.9419e-09/2.0,1.9419e-09*2.0,np.abs((1.9419e-09*2.0-1.9419e-09/2.0))/100.0], #bphys['eIh'] --> h.e_Ih --> 1.9419e-09
		[8.0775/2.0,8.0775*2.0,np.abs((8.0775*2.0-8.0775/2.0))/100.0], #bphys['kIh'] --> h.k_Ih --> 8.0775
		[-90.963*2.0,-90.963/2.0,np.abs((90.963*2.0-90.963/2.0))/100.0], #bphys['vhIh'] --> h.vh_Ih --> -90.963
		[-49.765*2.0,-49.765/2.0,np.abs((49.765*2.0-49.765/2.0))/100.0] #bphys['ehcnIh'] --> h.ehcn_Ih --> -49.765
			]
	
				
	use_seed = True
	targetdir = os.getcwd()+'/Results/'
	if use_seed:
		with open(targetdir+'MatingPool_reserve.pickle','rb') as f:
			seed = pickle.load(f)
		evol = Evolve(modelscript='RunL5.py',param_ranges=params,outputdir=targetdir,threshold_type='pool',score_threshold=0.05,generation_limit=4000,seed_generation=seed[0])
	else:
		evol = Evolve(modelscript='RunL5.py',param_ranges=params,outputdir=targetdir,threshold_type='pool',score_threshold=0.05,generation_limit=4000)
	evol.run()


