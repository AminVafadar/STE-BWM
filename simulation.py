"""
Script to generate an object for Amin's Spanning Tree Enumeration (STE) Best-Worst Method (BWM) extension.

This script is designed to run simulations not implementation, i.e. it will solve a set of problem sets not
be a fully-fledged program to interact with the users.

The script has two main parts:
1) A class for the problem set is defined alongside the methods for the commonly performed operations.
2) A method is implemented to define the procedure for running the simulations.

"""

# imports
import random
from string import ascii_lowercase
from sympy.combinatorics.graycode import GrayCode
from itertools import compress
import numpy as np
from scipy.stats.mstats import gmean
import pandas as pd
import datetime as dt


# define the problem set class
class SteBwmSimulation:
	
	def __init__(self, criteria_no, best_no, worst_no):
		"""
		Initialise the class with the necessary parameters and variables.
		:param criteria_no: int; the number of decision making criteria to use
		:param best_no: int; the number of criteria to draw to the best set
		:param worst_no: int; the number of criteria to draw to the worst set
		"""
		self.criteria_no = criteria_no
		self.best_no = best_no
		self.worst_no = worst_no
		# generate a list of letters denoting the decision making criteria
		self.criteria = [i for i in ascii_lowercase[0:criteria_no]]
		# create a set of k 'best' criteria
		self.theta = self.criteria[0:self.best_no]
		# create a set of k 'worst' criteria (from the choices excluding theta)
		self.gamma = self.criteria[self.best_no:self.best_no+self.worst_no]
		# randomly generate value for the worst set
		self.value_worst = random.randint(3, 9)
		
	def __str__(self):
		return 'This is an STE-BWM object initialised with the following settings:' + \
			   '\nNumber of DM Criteria: ' + str(self.criteria_no) + \
			   '\nNumber of best criteria: ' + str(self.best_no) + \
			   '\nNumber of worst criteria: ' + str(self.worst_no) + \
			   '\nDM Criteria: ' + ', '.join(self.criteria) + \
			   '\nTheta set: ' + ', '.join(self.theta) + \
			   '\nGamma set: ' + ', '.join(self.gamma) + \
			   '\nValue for the worst: ' + str(self.value_worst)
				
	def generate_bto_vectors(self):
		"""
		Method to generate best-to-others vectors for each best criterion. It operates on the attributes of the
		initialised class object.
		:return: dict; dictionary with dm criteria as keys, and best-to-others vectors (lists) as values.
		"""
		# initialise the dictionary to hold the results
		final_vectors = {}
		# loop over each best criterion to create a best-to-others vector
		for criterion in self.theta:
			single_vector = []
			# loop over all criteria to generate values for each
			for x in self.criteria:
				# if the element is in the set of best criteria, value is 1
				if x in self.theta:
					single_vector.append(1)
				# if the element is in the set of worst criteria, the value is the same as the value for the worst criterion
				elif x in self.gamma:
					single_vector.append(self.value_worst)
				# if it's neither of the above generate a random value > 1 but < value for the worst - 1
				else:
					single_vector.append(random.randint(2, self.value_worst - 1))
			# append that vector to the dictionary
			final_vectors[criterion] = single_vector
		# return the bto vectors
		return final_vectors
	
	def generate_otw_vectors(self):
		"""
		Method to generate others-to-worst vectors for each worst criterion. It operates on the attributes of the
		initialised class object.
		:return: dict; dictionary with dm criteria as keys, and others-to-worst vectors (lists) as values.
		"""
		# initialise the dictionary to hold the results
		final_vectors = {}
		# loop over each best criterion to create a best-to-others vector
		for criterion in self.gamma:
			single_vector = []
			# loop over all criteria to generate values for each
			for x in self.criteria:
				# if the element is in the set of best criteria, value is 1
				if x in self.theta:
					single_vector.append(self.value_worst)
				# if the element is in the set of worst criteria, the value is the same as the value for the worst criterion
				elif x in self.gamma:
					single_vector.append(1)
				# if it's neither of the above generate a random value > 2 but < value for the worst - 1
				else:
					single_vector.append(random.randint(2, self.value_worst - 1))
			# append that vector to the dictionary
			final_vectors[criterion] = single_vector
		# return the bto vectors
		return final_vectors
	
	def calculate_consistency(self, bto_vectors, otw_vectors):
		"""
		Method to calculate consistency values for each pair of best/worst vectors
		:param bto_vectors: dict; dictionary of best-to-others vectors
		:param otw_vectors: dict; dictionary of others-to-worst vectors
		:return: dict; dictionary of consistency ratios for each bto/otw pair
		"""
		# define the use as a divisor in all calculations
		value_worst_divisor = self.value_worst * self.value_worst - self.value_worst
		# dictionary to hold the results
		final_vectors = {}
		# loop over the pair of each bto and otw vectors
		for bto_key in bto_vectors.keys():
			for otw_key in otw_vectors.keys():
				consistency_vector = []
				# loop over the elements of each bto_vectors and calculate consistency
				for i in range(len(bto_vectors[bto_key])):
					consistency_vector.append(
						abs(bto_vectors[bto_key][i] * otw_vectors[otw_key][i] - self.value_worst) / value_worst_divisor
					)
				# append the consistency vector to the resulting dictionary; use criteria combination as key
				final_vectors[bto_key + otw_key] = consistency_vector
		# return the resulting dictionary
		return final_vectors
	
	def consistency_check(self, consistency_vectors):
		"""
		Check whether the parameters are below the consistency thresholds.
		:param consistency_vectors: vectors of consistencies calculated with 'calculate_consistency'
		:return: list of bools; T or F for each consistency check against the threshold.
		"""
		# dict of consistency thresholds
		consistency_thresholds_full = {
			3: {3: 0.1667, 4: 0.1121, 5: 0.1354, 6: 0.133, 7: 0.1294, 8: 0.1309, 9: 0.1359},
			4: {3: 0.1667, 4: 0.1529, 5: 0.1994, 6: 0.199, 7: 0.2457, 8: 0.2521, 9: 0.2681},
			5: {3: 0.1667, 4: 0.1898, 5: 0.2306, 6: 0.2643, 7: 0.2819, 8: 0.2958, 9: 0.3062},
			6: {3: 0.1667, 4: 0.2206, 5: 0.2546, 6: 0.3044, 7: 0.3029, 8: 0.3154, 9: 0.3337},
			7: {3: 0.1667, 4: 0.2527, 5: 0.2716, 6: 0.3144, 7: 0.3144, 8: 0.3408, 9: 0.3517},
			8: {3: 0.1667, 4: 0.2577, 5: 0.2844, 6: 0.3221, 7: 0.3251, 8: 0.362, 9: 0.362},
			9: {3: 0.1667, 4: 0.2683, 5: 0.296, 6: 0.3262, 7: 0.3403, 8: 0.3657, 9: 0.3662}
		}
		# return max from each consistency vector
		cr_maxes = {}
		for consistency_vector in consistency_vectors.keys():
			cr_maxes[consistency_vector] = max(consistency_vectors[consistency_vector])
		# check against the consistency thresholds
		final_checks = []
		consistency_thresholds = consistency_thresholds_full[self.criteria_no]
		for item in cr_maxes.items():
			# print('checking ', item[1], ' with ', consistency_thresholds[self.value_worst])
			final_checks.append(item[1] < consistency_thresholds[self.value_worst])
		# return the list of checks
		return final_checks
	
	def generate_trees(self):
		"""
		In the network trees criteria are represented as nodes (n), and the connections between them are edges (m).
		To construct a Minimum Spanning Tree (MST) we need n-1 edges.
		:return: list of potential trees represented as 0s and 1s (for edges).
		"""
		# generate Gray Codes for the number of edges (nodes * 2 - 3)
		temp = GrayCode(self.criteria_no*2-3)
		graycodes_list = list(temp.generate_gray())
		# count 1s in all Gray Codes (the number of edges between our nodes), and mark True those with n-1
		counted_list = []
		for code in graycodes_list:
			count = 0
			for i in code:
				if i == '1':
					count += 1
			# mark as True the codes with the required minimum number of edges
			if count == self.criteria_no - 1:
				counted_list.append(True)
			else:
				counted_list.append(False)
		# keep only Gray Codes which contain potential trees
		potential_trees = list(compress(graycodes_list, counted_list))
		# return the trees
		return potential_trees
	
	def create_edges(self, vector1, vector2):
		"""
		Helper function to create edges between the two vectors.
		We ignore key loops, and need only one edge between the keys of two vectors.
		:param vector1: list; bto or otw vector.
		:param vector2: list; bto or otw vector.
		:return: list of possible edges between the keys (nodes) of the two vectors.
		"""
		# loop over the two vectors to create all possible edges
		edges = []
		# first go over one of the vectors and create all edges except for the looped one
		for criterion in self.criteria:
			dict_key = vector1[0]
			if criterion != dict_key:
				edges.append(dict_key + criterion)
		# then go over the second vector and create all edges except for the looped one, and the one which already exists
		for criterion in self.criteria:
			dict_key = [vector2[0], vector1[0]]
			if criterion not in dict_key:
				edges.append(criterion + dict_key[0])
		return edges
	
	def generate_msts(self, vector1, vector2, potential_trees):
		"""
		Method to generate graphs of all Minimum Spanning Trees (MST) for two given vectors (bto and otw).
		:param vector1: dict; bto or otw vector.
		:param vector2: dict; bto or otw vector.
		:param potential_trees: list; list of potential candidate graphs to construct MSTs.
		:return: list of MSTs as a list of lists of edges and as a list of Gray Codes.
		"""
		# create edges
		# loop over the two vectors to create all possible edges
		edges = self.create_edges(vector1, vector2)
		# translate 1s from the Gray Codes into edges in graphs - return graphs in a form of list of edges
		possible_graphs = []
		for graycode in potential_trees:
			present_edges = []
			counter = 0
			for element in graycode:
				if element == '1':
					present_edges.append(edges[counter])
				counter += 1
			possible_graphs.append(present_edges)
		# identify disconnected graphs, i.e. with at least one node without an edge - they cannot be MSTs
		final_check = []
		for single_graph in possible_graphs:
			single_graph_joined = ''.join(single_graph)
			toappend = True  # assume it's present but revise it if a criterion not present in the string of edge pairs
			for criterion in self.criteria:
				if criterion not in single_graph_joined:
					toappend = False
			final_check.append(toappend)
		# keep only minimum spanning trees
		final_msts = list(compress(possible_graphs, final_check))
		# return them
		return final_msts
	
	def calculate_weights(self, best_vector, worst_vector, potential_trees, mean_type='arithmetic', ):
		"""
		Method to generate weights based on two vectors.
		:param potential_trees: list of values; previously generated all potential trees.
		:param mean_type: str; either arithmetic or geometric. Type of mean to be used for the weights.
		:param best_vector: dict; bto vector.
		:param worst_vector: dict; otw vector.
		:return: numpy array; weights for each of the criteria based on the two vectors supplied.
		"""
		# generate all of the minimum spanning trees
		msts = self.generate_msts(best_vector, worst_vector, potential_trees)
		# loop over each of the MST and calculate weights
		final_weights = []  # a list to hold the weights calculated for each MST
		for mst in msts:
			# start a test loop
			row_list = []
			for pair in mst:
				# create a row to modify based on the edge pair
				start_row = [0] * self.criteria_no
				# get the original indexes of the edge pair
				index_first_el = self.criteria.index(pair[0])
				index_second_el = self.criteria.index(pair[1])
				# modify the row based on the values from the edge pair - different scenarios for best and worst
				if pair[0] == best_vector[0]:
					# get the negated weight
					weight_negated = best_vector[1][index_second_el] * -1
					# insert into the original row
					start_row[index_second_el] = weight_negated
					start_row[index_first_el] = 1
				else:
					# get the negated weight
					weight_negated = worst_vector[1][index_first_el] * -1
					# insert into the original row
					start_row[index_second_el] = weight_negated
					start_row[index_first_el] = 1
				# add
				row_list.append(start_row)
			# add a row of 1s to prepare a matrix for solving the set of linear equations
			row_list.append([1] * self.criteria_no)
			# turn the list of lists into a numpy array and solve the linear equations
			a = np.array(row_list)
			b = np.array([0] * (self.criteria_no-1) + [1])
			x = np.linalg.solve(a, b)
			# add the weights for this MST to the final list
			final_weights.append(list(x))
		# get the total weights for the vector pair - mean of all the weights
		if mean_type == 'arithmetic':
			results = np.mean(final_weights, 0)
		elif mean_type == 'geometric':
			results = gmean(final_weights)
		else:
			results = 'Wrong type of mean specified.'
		# return the array with the results
		return results


# outside method for generating potential trees - this can speed up the simulation
def generate_trees(criteria_no):
	"""
	In the network trees criteria are represented as nodes (n), and the connections between them are edges (m).
	To construct a Minimum Spanning Tree (MST) we need n-1 edges.
	:type criteria_no: int; number of criteria to calculate potential edges.
	:return: a list of potential spanning trees, represented as 0s and 1s.
	"""
	# generate Gray Codes for the number of edges (nodes * 2 - 3)
	temp = GrayCode(criteria_no*2-3)
	graycodes_list = list(temp.generate_gray())
	# count 1s in all Gray Codes (the number of edges between our nodes), and mark True those with n-1
	counted_list = []
	for code in graycodes_list:
		count = 0
		for i in code:
			if i == '1':
				count += 1
		# mark as True the codes with the required minimum number of edges
		if count == criteria_no - 1:
			counted_list.append(True)
		else:
			counted_list.append(False)
	# keep only Gray Codes which contain potential trees
	potential_trees = list(compress(graycodes_list, counted_list))
	# return the trees
	return potential_trees


# function to run the simulation
def run_simulation(list_of_criteria_numbers, number_of_replications, mean_to_use, output_directory):
	"""
	Final function which puts the whole simulation procedure into one method.
	For each combination of DM criteria, best and worst vectors, object is generated and replicated a specified
	amount of times. If it fails the consistency check then it starts again.
	
	:param list_of_criteria_numbers: list of integers; each integer in the list specifies the number of decision making
	criteria used for the specific simulation. There will be two files with results for each number specified.
	:param number_of_replications: int; number of times each 'scenario' is to be replicated. A scenario is a unique
	combination of number of decision making criteria + number of best vectors + number of worst vectors, e.g.
	6 decision making criteria, 1 best vector and 2 worst vectors.
	:param mean_to_use: str; there are two options 'arithmetic' or 'geometric'. This choice is used throughout for
	three mean calculations, i.e. 1) mean of weights of individual spanning trees of two vectors;
	2) mean of weights of all vector (best/worst) pairs; 3) mean of weights of all simulations in a scenario.
	:param output_directory: path; system path to the directory to which you want to write the results.
	:return: csv files; results in detailed and aggregated form: for each replication and for each scenario.
	"""
	# first loop over each number of criteria
	for criteria_number in list_of_criteria_numbers:
		# create two dataframes - one with very detailed results, and one on a more aggregated level
		detailed_results = pd.DataFrame(
			columns=
			['number_of_criteria', 'number_of_best', 'number_of_worst', 'simulation_number'] +
			[i for i in ascii_lowercase[0:criteria_number]]
		)
		aggregated_results = pd.DataFrame(
			columns=
			['number_of_criteria', 'number_of_best', 'number_of_worst', 'number_of_simulations'] +
			[i for i in ascii_lowercase[0:criteria_number]]
		)
		# create potential spanning trees for this iteration - this can be reused in each weight calculations
		potential_trees = generate_trees(criteria_number)
		# for each criteria number loop over all possibilities of best_no and worst_no combinations
		for number_of_best in range(1, criteria_number):
			for number_of_worst in range(1, criteria_number):
				if number_of_best + number_of_worst > criteria_number:
					continue
				simulation_results = []
				print('Criteria number', criteria_number)
				print('Number of best', number_of_best)
				print('Number of worst', number_of_worst)
				# while loop equal to the number of replications, increasing a replication when consistency check is passed
				simulation_number = 0
				while simulation_number < number_of_replications:
					# initialise the object
					test_bwm = SteBwmSimulation(criteria_no=criteria_number,
												best_no=number_of_best,
												worst_no=number_of_worst)
					# generate bto and otw vectors
					bto = test_bwm.generate_bto_vectors()
					otw = test_bwm.generate_otw_vectors()
					# consistency check
					consistency = test_bwm.calculate_consistency(bto, otw)
					consistency_checks = test_bwm.consistency_check(consistency)
					if False in consistency_checks:
						continue
					else:
						# calculate weights for each vector pair (of best and worst)
						resulting_weights = []
						for each_bto in bto.items():
							for each_otw in otw.items():
								resulting_weights.append(list(test_bwm.calculate_weights(each_bto,
																						 each_otw,
																						 potential_trees,
																						 mean_type=mean_to_use)))
						# aggregate the results across each pair
						if mean_to_use == 'arithmetic':
							simulation_weights = np.mean(resulting_weights, 0)
						elif mean_to_use == 'geometric':
							simulation_weights = gmean(resulting_weights)
						else:
							simulation_weights = 'Wrong type of mean specified.'
						# add the results to the detailed dataset
						detailed_results = detailed_results.append(
							pd.Series([criteria_number, number_of_best, number_of_worst, simulation_number] +
									  list(simulation_weights),
									  index=detailed_results.columns),
							ignore_index=True
						)
						simulation_results.append(list(simulation_weights))
						# increase the simulation number
						simulation_number += 1
				# get the mean of the results from the particular scenario and add it to the aggregated results
				if mean_to_use == 'arithmetic':
					simulation_summary_weights = np.mean(simulation_results, 0)
				elif mean_to_use == 'geometric':
					simulation_summary_weights = gmean(simulation_results)
				# add the results to the aggregated dataset
				aggregated_results = aggregated_results.append(
					pd.Series(
						[criteria_number, number_of_best, number_of_worst, simulation_number] +
						list(simulation_summary_weights),
						index=aggregated_results.columns),
					ignore_index=True
				)
		# save the files
		detailed_results.to_csv("{}/detailed_results_{}_dm_criteria_{}_mean_{}_replications_{}.csv".format(
			output_directory, criteria_number, mean_to_use, number_of_replications, dt.date.today().strftime("%Y%m%d")))
		aggregated_results.to_csv("{}/aggregated_results_{}_dm_criteria_{}_mean_{}_replications_{}.csv".format(
			output_directory, criteria_number, mean_to_use, number_of_replications, dt.date.today().strftime("%Y%m%d")))
