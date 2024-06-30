"""
A script for running the simulations of STE BWM defined in 'simulation' script.

Parameters to specify:
list_of_criteria_numbers: list of integers; each integer in the list specifies the number of decision making criteria
						  used for the specific simulation.
						  There will be two files with results for each number specified.
						  Any combination of the numbers can be specified, e.g. [6], [3, 9], [3, 4, 5, 6], etc.
number_of_replications: int; number of times each 'scenario' is to be replicated. A scenario is a unique combination of
						number of decision making criteria + number of best vectors + number of worst vectors, e.g.
						6 decision making criteria, 1 best vector and 2 worst vectors.
						Any number can be specified but note that thousands of replications might take a substantial
						amount of time to run.
mean_to_use: str; there are two options 'arithmetic' or 'geometric'. This choice is used throughout for three mean
				  calculations, i.e. 1) mean of weights of individual spanning trees of two vectors;
				  					 2) mean of weights of all vector (best/worst) pairs;
				  					 3) mean of weights of all simulations in a scenario.
output_directory: path; system path to the directory to which you want to write the results.
"""

# imports
from simulation import run_simulation

# PLEASE SPECIFY THE PROBLEM SET SIMULATION CRITERIA
list_of_criteria_numbers = [3]
number_of_replications = 10
mean_to_use = 'geometric'
output_directory = '/Users/amin/Desktop/simulation_results'

# simulation to run
run_simulation(
	list_of_criteria_numbers,
	number_of_replications,
	mean_to_use,
	output_directory
)
