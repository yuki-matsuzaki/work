#coding:utf-8
import csv
import sys
import argparse
import string
import math
import numpy as np
from numpy.random import *
# from tqdm import tqdm
import time

global ptype
ptype = {"":0, "cart":1, "cart_form":2, "category":3, 
		"conversion":4, "item":5, "login":6, "ranking":7, 
		"registration":8, "registration_form":9, "review":10, 
		"sale":11, "search":12, "top":13}
global K
K = len(ptype)

global DBL_MIN
DBL_MIN = sys.float_info.min

global log_DBL_MIN
log_DBL_MIN = math.log10(DBL_MIN)

def load_data(fname):
	out = []
	fp1 = open(fname)

	reader = csv.reader(fp1)
	header = None

	usid_list = []
	page_type_list = []
	is_cv_list = []

	# データを流す
	#
	#流すデータはusidでソートされたもの
	pv = 0
	i=0
	for row in reader :
		if header == None:
			header = row
			continue
		if row[5] == "_custom_":
			continue
		usid_list.append(row[1])
		page_type_list.append(row[5])
		is_cv_list.append(row[9])
		# is_ctr_list.append(row[10])
		
	usid_list.append("EOF")
	page_type_list.append("EOF")
	is_cv_list.append("EOF")

	new_usid_list = []
	new_page_type_list = []
	new_is_cv_list = []

	# delete pv = 1
	pv = 0 
	for i in range(0, len(usid_list)-2):
		new_usid_list.append(usid_list[i])
		new_page_type_list.append(page_type_list[i])
		new_is_cv_list.append(is_cv_list[i])
		pv += 1

		if usid_list[i] != usid_list[i+1]:
			if pv > 1:
				pv = 0
			else :
				del new_usid_list[-1]
				del new_page_type_list[-1]
				del new_is_cv_list[-1]

	# for dict of uniq user
	usid = 0
	usid_num_dict = {}
	for user in new_usid_list:
		if user not in usid_num_dict:
			usid_num_dict[user] = usid
			# print user
			# print usid_num_dict[user]
			# x = raw_input()
			usid += 1
	# print usid
	# print len(usid_num_dict)
	# print usid_num_dict["560c69c8bb0e0e7766cca076"]
	# x = raw_input()
	out.append(new_usid_list)
	out.append(new_page_type_list)
	out.append(new_is_cv_list)
	out.append(usid_num_dict)
	return out

def make_learing_data(usid_list, page_type_list, usid_num_dict, I):
	# for initial matrix which is consists of binary values
	# and n_i.j.k which is number of trans. j to k

	np.initial_matrix = [[0 for i in range(K)] for j in range(I)]
	ptype_seq = []
	# np.N = [0 for i in range(I)]
	np.N = [[[0 for i in range(K)] for i in range(K)] for i in range(I)]

	for i in range(0, len(usid_list)-1):
		ptype_seq.append(ptype[page_type_list[i]])
		if usid_list[i] != usid_list[i+1]:

			np.initial_matrix[usid_num_dict[usid_list[i]]][ptype_seq[0]] = 1

			for j in range(0, len(ptype_seq)-2):
				np.N[usid_num_dict[usid_list[i]]][ptype_seq[j]][ptype_seq[j+1]] += 1
			
			ptype_seq = []
		
	# for i in range(len(usid_list)-1):
	# 	print np.N[i]
	# 	x = raw_input()

	out = []
	out.append(np.initial_matrix)
	out.append(np.N)
	return out

def initialize_paras():
	# np.a = rand(S, K, K)
	np.a = []
	np.a_row = []
	np.a_matrix = []

	for s in range(S):
		for k in range(K):
			a_seed = rand(K)
			for item in a_seed:
				np.a_row.append(item/sum(a_seed))
			np.a_matrix.append(np.a_row)
			np.a_row = []
		np.a.append(np.a_matrix)
		np.a_matrix = []
	# for s in range(S):
	# 	for j in range(K):
	# 		print sum(np.a[s][j])
	# 		x = raw_input()
	np.pi =[]
	# pi_seed = rand(S)
	for i in range(S):
		np.pi.append(1/float(S))
		
	np.Lambda = []
	np.Lambda_row = []

	for k in range(K):
		Lambda_seed = rand(S)
		for item in Lambda_seed:
			np.Lambda_row.append(item/sum(Lambda_seed))
		np.Lambda.append(np.Lambda_row)
		np.Lambda_row = []
	
	out = []
	out.append(np.a)
	out.append(np.Lambda)
	out.append(np.pi)
	return out


def E_step(a, pi, Lambda, N, initial_matrix, I):
	# exponent part of a and lambda
	np.exponent_a = [[0.0 for i in range(S)] for i in range(I)]
	np.exponent_lambda = [[0.0 for i in range(S)] for i in range(I)]
	np.exponent = [[0.0 for i in range(S)] for i in range(I)]
	np.denominator = [0.0 for i in range(I)]

	for i in range(I):
		for s in range(S):
			for j in range(K):
				# np.exponent_lambda[i][s] += initial_matrix[i][j] * math.log10(Lambda[j][s])
				if Lambda[j][s] <= DBL_MIN:
					np.exponent_lambda[i][s] += initial_matrix[i][j]*log_DBL_MIN
				else:
					np.exponent_lambda[i][s] += initial_matrix[i][j]*math.log10(Lambda[j][s])
				
				for k in range(K):
					if a[s][j][k] <= DBL_MIN:
						np.exponent_a[i][s] += N[i][j][k]*log_DBL_MIN
					else:
						np.exponent_a[i][s] += N[i][j][k]*math.log10(a[s][j][k])

			if np.exponent_a[i][s] < log_DBL_MIN:
				np.exponent_a[i][s] = log_DBL_MIN
			if np.exponent_lambda[i][s] < log_DBL_MIN:
				np.exponent_lambda[i][s] = log_DBL_MIN
			if np.exponent_a[i][s] + np.exponent_lambda[i][s] < log_DBL_MIN:
				np.exponent[i][s] = log_DBL_MIN
			else:
				np.exponent[i][s] = np.exponent_a[i][s] + np.exponent_lambda[i][s]
			np.denominator[i] += pi[s]*10**np.exponent[i][s]
		# time.sleep(1)

	# for outputs
	np.alpha = [[0.0 for i in range(S)] for i in range(I)]
	for i in range(I):
		for s in range(S):
			np.alpha[i][s] = (pi[s]*10**np.exponent[i][s])/np.denominator[i]

	# print np.alpha
	return np.alpha
	

def M_step(alpha, N, initial_matrix, I):
	np.a = [[[0.0 for i in range(K)] for i in range(K)] for i in range(S)] 
	np.pi = [0.0 for i in range(S)]
	np.Lambda = [[0.0 for i in range(S)] for i in range(K)]
	np.numerator = 0.0
	np.pi_sum = 0.0
	# iteration for pi and lambda
	for j in range(K):	
		for s in range(S):
			for i in range(I):
				np.pi_sum += alpha[i][s]
				# for numerator 
				np.numerator += alpha[i][s]*initial_matrix[i][j]
			
			# calculate
			np.Lambda[j][s] = np.numerator/np.pi_sum
			np.pi[s] = np.pi_sum/I
			# initialize
			np.numerator = 0.0
			np.pi_sum = 0.0

		# time.sleep(1)

	# iteration for a
	np.alpha_dsum = 0.0
	np.alpha_nsum = 0.0

	for s in range(S):
		for k in range(K):
			for j in range(K):
				
				for r in range(K):
					for i in range(I):
						np.alpha_dsum += alpha[i][s]*N[i][j][r]
				for i in range(I):
					np.alpha_nsum += alpha[i][s]*N[i][j][k]
			
				# calculate
				if np.alpha_dsum == 0.0:
					np.a[s][j][k] = 0.0	
			
				else :
					np.a[s][j][k] = np.alpha_nsum/np.alpha_dsum
				# initialize
				np.alpha_dsum = 0.0
				np.alpha_nsum = 0.0


	out = []
	out.append(np.a)
	out.append(np.pi)
	out.append(np.Lambda)

	return out

def log_likelyhood(pi, Lambda, a, initial_matrix, N, I):

	np.exponent_a = [[0.0 for i in range(S)] for i in range(I)]
	np.exponent_lambda = [[0.0 for i in range(S)] for i in range(I)]
	np.exponent = [[0.0 for i in range(S)] for i in range(I)]
	np.denominator = [0.0 for i in range(I)]

	LL = 0.0
	SUM = [0.0 for i in range(I)]
	
 	for i in range(I):
		for s in range(S):
			for j in range(K):
				if Lambda[j][s] <= DBL_MIN:
					np.exponent_lambda[i][s] += initial_matrix[i][j]*log_DBL_MIN
				else:
					np.exponent_lambda[i][s] += initial_matrix[i][j]*math.log10(Lambda[j][s])
				for k in range(K):
					if a[s][j][k] <= DBL_MIN:
						np.exponent_a[i][s] += N[i][j][k]*log_DBL_MIN
					else:
						#if math.log10(a[s][j][s]) > 0:
						#	print "math.log10(a[", s, "][", j, "][", s, "]) = ", math.log10(a[s][j][s])
						
						np.exponent_a[i][s] += N[i][j][k]*math.log10(a[s][j][k])

			if np.exponent_a[i][s] < log_DBL_MIN:
				np.exponent_a[i][s] = log_DBL_MIN
			if np.exponent_lambda[i][s] < log_DBL_MIN:
				np.exponent_lambda[i][s] = log_DBL_MIN
			if np.exponent_a[i][s] + np.exponent_lambda[i][s] < log_DBL_MIN:
				np.exponent[i][s] = log_DBL_MIN
			else:
				np.exponent[i][s] = np.exponent_a[i][s]+np.exponent_lambda[i][s]

			if np.exponent_a[i][s] > 0 or np.exponent_lambda[i][s] > 0:
				print "np.exponent_a[", i, "][", s, "] = ", np.exponent_a[i][s]
				print "np.exponent_lambda[", i, "][", s, "] = ", np.exponent_lambda[i][s]
				x = raw_input()

			# exponent part 
			# math.log10(pi[s])+np.exponent[i][s]
			SUM[i] += pow(10.0, math.log10(pi[s])+np.exponent[i][s])
			# print "SUM[", i, "]= ", SUM[i]
			# x = raw_input()

		LL += math.log10(SUM[i])
		# print "LL"
		# print LL

	return LL

def main(fname):

	#loading data
	usid_list = []
	page_type_list = []
	is_cv_list = []
	usid_num_dict = {}

	print "loading data..."
	
	(usid_list, page_type_list, is_cv_list, usid_num_dict)	= load_data(fname)
	
	# number of usid
	I = len(usid_num_dict)
	
	print "making learning data..."
	np.initial_matrix = []
	np.N = []
	(np.initial_matrix, np.N) = make_learing_data(usid_list, page_type_list, usid_num_dict, I)

	print "estimating parameters..."

	# initialize parameters
	np.a = []
	np.Lambda = []
	np.pi = []

	(np.a, np.Lambda, np.pi) = initialize_paras()

	# posterior probability of latent segment
	np.alpha = []
	
	LL_prev = -sys.maxint - 1
	MAX_iter = 1000
	convergence = 1.0e-3

	for i in range(MAX_iter):
		np.alpha = E_step(np.a, np.pi, np.Lambda, np.N, np.initial_matrix, I)
		(np.a, np.pi, np.Lambda) = M_step(np.alpha, np.N, np.initial_matrix, I)

		LL = log_likelyhood(np.pi, np.Lambda, np.a, np.initial_matrix, np.N, I)

		print "iteration",i
		print "log_likelyhood = ", LL
		print "Improvement rate = ", abs(LL - LL_prev)/abs(LL_prev)
		if LL < LL_prev:
			print " LL < LL_prev"
			break
		if abs(LL - LL_prev)/abs(LL_prev) < convergence: 
			break
		else :
			LL_prev = LL

	fname = fname[0:-4]

	fp_a = open("a_%s_s=%s.csv" % (fname, str(S)), "wb")
	csv_writer = csv.writer(fp_a)
	for item in np.a:
		csv_writer.writerows(item)

	fp_pi = open("pi_%s_s=%s.csv" % (fname, str(S)), "wb")
	csv_writer = csv.writer(fp_pi)
	csv_writer.writerow(np.pi)

	fp_lambda = open("lambda_%s_s=%s.csv" % (fname, str(S)), "wb")
	csv_writer = csv.writer(fp_lambda)
	csv_writer.writerows(np.Lambda)
	print "done!"

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='model of dias. input filename n number of latent classes...')
	parser.add_argument('ifilename')
	parser.add_argument('latent_class')
	
	args = parser.parse_args()
	fname = args.ifilename
	global S
	S = int(args.latent_class)

	main(fname)