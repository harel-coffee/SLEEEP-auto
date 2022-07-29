# 
# 3compute_robustness.py
# this code computesd ICCs for each anatomical area over each tool
#
############################################################################### 
import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))
from tqdm import tqdm
import pandas as pd
import pingouin as pg
import numpy as np
import multiprocessing as mp
from utils import pickle_in, pickle_out, get_root, fisher_mean


def get_icc_df_chunk(column, tool):
	read_df = dataframe_collection[tool]
	return_df = pd.DataFrame(columns=['Scan', 'Tool', 'Score'])
	for timepoint in range(1, 5):
		tmp_df = pd.DataFrame(columns=['Scan', 'Tool', 'Score'])
		tmp_df['Scan'] = list(read_df.index)
		tmp_df['Tool'] = [tool] * len(read_df)
		tmp_df['Score'] = list(read_df[str(timepoint) + column].values)
		return_df = return_df.append(tmp_df)
	return return_df


def get_rmae_df_chunk(column, tool):
	read_df = dataframe_collection[tool]
	df = pd.DataFrame(columns=['Scan', 'Tool', 'Score1', 'Score2'])
	df['Scan'] = list(read_df.index)
	df['Tool'] = [tool] * len(read_df)
	df['Score1'] = list(read_df[str(1) + column].values)
	df['Score2'] = list(read_df[str(2) + column].values)
	return df


def get_icc_df_chunk2(column, tool):
	read_df = dataframe_collection[tool]
	df = pd.DataFrame(columns=['Scan', 'Tool', 'Score'])
	df['Scan'] = list(read_df.index) * 2
	df['Tool'] = ['TP1' + tool ] * len(read_df) + ['TP2' + tool ] * len(read_df)
	df['Score'] = list(read_df['1' + column]) + list(read_df['2' + column])
	return df


def compute_icc(column, return_dict):
	icc_df = pd.DataFrame(columns=['Scan', 'Tool', 'Score'])
	for tool in tools:
		icc_df = icc_df.append(get_icc_df_chunk(column, tool))
	icc = pg.intraclass_corr(data=icc_df, targets='Scan', raters='Tool', ratings='Score', nan_policy='omit')
	return_dict[column] = icc
	return


def compute_icc2(column, return_dict):
	rmae_df = pd.DataFrame(columns=['Scan', 'Tool', 'Score'])
	icc_list = []
	for tool in tools:
		rmae_df = rmae_df.append(get_icc_df_chunk2(column, tool))
		icc = pg.intraclass_corr(data=rmae_df, targets='Scan', raters='Tool', ratings='Score', nan_policy='omit')
		icc_list.append(icc['ICC'][2])
	return_dict[column] = fisher_mean(icc_list)
	return


def compute_rmae(column, return_dict):
	rmae_df = pd.DataFrame(columns=['Scan', 'Tool', 'Score1', 'Score2'])
	for tool in tools:
		rmae_df = rmae_df.append(get_rmae_df_chunk(column, tool))

	r = np.nanmean(list(rmae_df['Score1']) + list(rmae_df['Score2']))
	ae = np.abs(rmae_df['Score1'] - rmae_df['Score2'])
	mae = np.nanmean(ae)
	rmae = mae/r

	return_dict[column] = rmae
	breakpoint()
	return

def chunks(l, n):
	for i in range(0, len(l), n):
		yield l[i:i + n]

if __name__ == '__main__':
	#task = 'icc'
	#task = 'rmae'
	task = 'icc2'
	#vol = 'MEMPRAGE'
	vol = 'DTI'
	#vol = 'BRAIN'
	PARALLEL = False
	n_cores = 10
	# Source
	data_frame_name = 'raw_radiomic_dataframe_' + vol + '_'
	# Target
	target_file = get_root('scripts', 's7_Radiomics_analysisAndVisualization', 'robustness_' + task + '_' + vol + '_results.pkl')

	# path variables
	tools = ['vuno', 'samseg', 'freesurfer', 'fastsurfer']

	# Get appropriate method for task
	if task == 'icc':  # ICC over raters
		compute = compute_icc
	elif task == 'rmae':
		compute = compute_rmae
	elif task == 'icc2':  # ICC between time points
		compute = compute_icc2
	else:
		raise ValueError('Unrecognized task "' + task + '"')

	# to prevent repetitive I/O we load each dataframe once and store it im memory
	dataframe_collection = {}

	# create a nested list with columns from each radiomics file
	nested_cols_list = []
	for tool in tools:
		df = pickle_in(get_root('data', 'radiomics_dataframes', data_frame_name + tool + '.pkl'))
		df = df.drop('is_deprived', axis=1)
		dataframe_collection[tool] = df
		df_cols = list(set([column[1:] for column in df]))
		nested_cols_list.append(df_cols)
	# find mutual columns across all tools
	columns = list(set.intersection(*map(set, nested_cols_list)))
	if PARALLEL and len(columns) < mp.cpu_count():  # Parallel process
		manager = mp.Manager()
		return_dict = manager.dict()
		jobs = []
		for i, column in enumerate(columns):
			p = mp.Process(target=compute, args=(column, return_dict))
			jobs.append(p)
			p.start()

		for proc in jobs:
			proc.join()
	else:  # Non-parallel process
		results = {}
		for column in tqdm(columns):
			compute(column, results)
	pickle_out(results, target_file)
	print('Successfully stored results to:', target_file)



