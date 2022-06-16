from ass_frame import statistics as st
import os
import sklearn.metrics, scikitplot.metrics
import numpy as np
import matplotlib.pyplot as plt

data_path = 'data'

summary = open('summary.txt', 'w')

# original test data
sef = st(os.path.join(data_path, 'sef_pre.csv'), 'Seferbekov Original')
VM = st(os.path.join(data_path, 'VM_pre.csv'), 'TeamVM Original')
ntech = st(os.path.join(data_path, 'ntech_pre.csv'), 'Ntech Original')

# processed test data
sef_pro = st(os.path.join(data_path, 'sef_pro.csv'), 'Seferbekov Processed')
VM_pro = st(os.path.join(data_path, 'VM_pro.csv'), 'TeamVM Processed')
ntech_pro = st(os.path.join(data_path, 'ntech_pro.csv'), 'Ntech Processed')

# ground truth labels
GT = st(os.path.join(data_path, 'GT.csv'))

# original test data (labelized with best threshold)
l_sef = sef.assign_labels(sef.find_best_threshold(GT), inplace=False)
l_ntech = ntech.assign_labels(ntech.find_best_threshold(GT), inplace=False)
l_VM = VM.assign_labels(VM.find_best_threshold(GT), inplace=False)

# processed test data (labelized with best threshold)
l_sef_pro = sef_pro.assign_labels(sef_pro.find_best_threshold(GT), inplace=False)
l_ntech_pro = ntech_pro.assign_labels(ntech_pro.find_best_threshold(GT), inplace=False)
l_VM_pro = VM_pro.assign_labels(VM_pro.find_best_threshold(GT), inplace=False)

# original test data (labelized with threshold = 0.5)
l_sef_half = sef.assign_labels(0.5, inplace=False)
l_ntech_half = ntech.assign_labels(0.5, inplace=False)
l_VM_half = VM.assign_labels(0.5, inplace=False)

# processed test data (labelized threshold = 0.5)
l_sef_pro_half = sef_pro.assign_labels(0.5, inplace=False)
l_ntech_pro_half = ntech_pro.assign_labels(0.5, inplace=False)
l_VM_pro_half = VM_pro.assign_labels(0.5, inplace=False)

number_real_vids = sum([1 for vid in GT.data if GT.data[vid] == 1.0])
number_fake_vids = len(GT.data) - number_real_vids

def t(n=5):
	return n * '\t'

def save_confusion_matrix():


	summary.write(f"""Confusion Matrix
								False Positive Rate\t\tTrue Positive Rate\t\tFalse Negative Rate\t\tTrue Negative Rate				
Seferbekov Original Dataset:	{l_sef_half.fp_rate(GT):.3f}{t()}{l_sef_half.tp_rate(GT):.3f}{t()}{l_sef_half.fn_rate(GT):.3f}{t()}{l_sef_half.tn_rate(GT):.3f}
Ntech Original Dataset			{l_ntech_half.fp_rate(GT):.3f}{t()}{l_ntech_half.tp_rate(GT):.3f}{t()}{l_ntech_half.fn_rate(GT):.3f}{t()}{l_ntech_half.tn_rate(GT):.3f}
TeamVM Original Dataset:		{l_VM_half.fp_rate(GT):.3f}{t()}{l_VM_half.tp_rate(GT):.3f}{t()}{l_VM_half.fn_rate(GT):.3f}{t()}{l_VM_half.tn_rate(GT):.3f}

Seferbekov Processed Dataset:	{l_sef_pro_half.fp_rate(GT):.3f}{t()}{l_sef_pro_half.tp_rate(GT):.3f}{t()}{l_sef_pro_half.fn_rate(GT):.3f}{t()}{l_sef_pro_half.tn_rate(GT):.3f}
Ntech Processed Dataset:		{l_ntech_pro_half.fp_rate(GT):.3f}{t()}{l_ntech_pro_half.tp_rate(GT):.3f}{t()}{l_ntech_pro_half.fn_rate(GT):.3f}{t()}{l_ntech_pro_half.tn_rate(GT):.3f}
TeamVM Processed Dataset:		{l_VM_pro_half.fp_rate(GT):.3f}{t()}{l_VM_pro_half.tp_rate(GT):.3f}{t()}{l_VM_pro_half.fn_rate(GT):.3f}{t()}{l_VM_pro_half.tn_rate(GT):.3f}
--------------------------------------------------------------------------
--------------------------------------------------------------------------

""")



def save_acc():


	summary.write(f"""BEST POSSIBLE ACCURACY

Seferbekov Original Dataset:	{l_sef.acc(GT):.3f}
Ntech Original Dataset			{l_ntech.acc(GT):.3f}
TeamVM Original Dataset:		{l_VM.acc(GT):.3f}

Seferbekov Processed Dataset:	{l_sef_pro.acc(GT):.3f}
Ntech Processed Dataset:		{l_ntech_pro.acc(GT):.3f}
TeamVM Processed Dataset:		{l_VM_pro.acc(GT):.3f}
--------------------------------------------------------------------------
--------------------------------------------------------------------------

""")


	summary.write(f"""ACCURACY WITH THRESHOLD = 0.5

Seferbekov Original Dataset:	{l_sef_half.acc(GT):.3f}
Ntech Original Dataset			{l_ntech_half.acc(GT):.3f}
TeamVM Original Dataset:		{l_VM_half.acc(GT):.3f}

Seferbekov Processed Dataset:	{l_sef_pro_half.acc(GT):.3f}
Ntech Processed Dataset:		{l_ntech_pro_half.acc(GT):.3f}
TeamVM Processed Dataset:		{l_VM_pro_half.acc(GT):.3f}
--------------------------------------------------------------------------
--------------------------------------------------------------------------

""")


def save_logloss():

	logloss_sef = sef.get_logloss(GT)
	logloss_VM = VM.get_logloss(GT)
	logloss_ntech = ntech.get_logloss(GT)

	logloss_sef_pro = sef_pro.get_logloss(GT)
	logloss_VM_pro = VM_pro.get_logloss(GT)
	logloss_ntech_pro = ntech_pro.get_logloss(GT)


	summary.write(f"""LOGLOSS

Seferbekov Original Dataset:	{logloss_sef:.3f}
Ntech Original Dataset			{logloss_ntech:.3f}
TeamVM Original Dataset:		{logloss_VM:.3f}

Seferbekov Processed Dataset:	{logloss_sef_pro:.3f}
Ntech Processed Dataset:		{logloss_ntech_pro:.3f}
TeamVM Processed Dataset:		{logloss_VM_pro:.3f}
--------------------------------------------------------------------------
--------------------------------------------------------------------------

""")


def save_mean_var():


	summary.write(f"""
MEAN SCORES

Seferbekov Original Dataset:	{sef.mean():.3f}
Ntech Original Dataset			{ntech.mean():.3f}
TeamVM Original Dataset:		{VM.mean():.3f}

Seferbekov Processed Dataset:	{sef_pro.mean():.3f}
Ntech Processed Dataset:		{ntech_pro.mean():.3f}
TeamVM Processed Dataset:		{VM_pro.mean():.3f}
--------------------------------------------------------------------------
--------------------------------------------------------------------------
""")


	summary.write(f"""
VARIANCE

Seferbekov Original Dataset:	{sef.var():.3f}
Ntech Original Dataset			{ntech.var():.3f}
TeamVM Original Dataset:		{VM.var():.3f}

Seferbekov Processed Dataset:	{sef_pro.var():.3f}
Ntech Processed Dataset:		{ntech_pro.var():.3f}
TeamVM Processed Dataset:		{VM_pro.var():.3f}
--------------------------------------------------------------------------
--------------------------------------------------------------------------

""")


def save_f_scores(betas, alpha):


	summary.write(f"""Confusion Matrix
								F{betas[0]} Score\t\tF{betas[1]} Score\t\tF{betas[2]} Score\t\tWeighted Precision (alpha={alpha})\t\tRecall		
Seferbekov Original Dataset:	{l_sef_half.f_score(GT, beta=betas[0]):.3f}{t(3)}{l_sef_half.f_score(GT, beta=betas[1]):.3f}{t(3)}{l_sef_half.f_score(GT, beta=betas[2]):.3f}{t(3)}{l_sef_half.weighted_precision(GT, alpha):.3f}{t(3)}{l_sef_half.recall(GT):.3f}
Ntech Original Dataset			{l_ntech_half.f_score(GT, beta=betas[0]):.3f}{t(3)}{l_ntech_half.f_score(GT, beta=betas[1]):.3f}{t(3)}{l_ntech_half.f_score(GT, beta=betas[2]):.3f}{t(3)}{l_ntech_half.weighted_precision(GT, alpha):.3f}{t(3)}{l_ntech_half.recall(GT):.3f}
TeamVM Original Dataset:		{l_VM_half.f_score(GT, beta=betas[0]):.3f}{t(3)}{l_VM_half.f_score(GT, beta=betas[1]):.3f}{t(3)}{l_VM_half.f_score(GT, beta=betas[2]):.3f}{t(3)}{l_VM_half.weighted_precision(GT, alpha):.3f}{t(3)}{l_VM_half.recall(GT):.3f}

Seferbekov Processed Dataset:	{l_sef_pro_half.f_score(GT, beta=betas[0]):.3f}{t(3)}{l_sef_pro_half.f_score(GT, beta=betas[1]):.3f}{t(3)}{l_sef_pro_half.f_score(GT, beta=betas[2]):.3f}{t(3)}{l_sef_pro_half.weighted_precision(GT, alpha):.3f}{t(3)}{l_sef_pro_half.recall(GT):.3f}
Ntech Processed Dataset:		{l_ntech_pro_half.f_score(GT, beta=betas[0]):.3f}{t(3)}{l_ntech_pro_half.f_score(GT, beta=betas[1]):.3f}{t(3)}{l_ntech_pro_half.f_score(GT, beta=betas[2]):.3f}{t(3)}{l_ntech_pro_half.weighted_precision(GT, alpha):.3f}{t(3)}{l_ntech_pro_half.recall(GT):.3f}
TeamVM Processed Dataset:		{l_VM_pro_half.f_score(GT, beta=betas[0]):.3f}{t(3)}{l_VM_pro_half.f_score(GT, beta=betas[1]):.3f}{t(3)}{l_VM_pro_half.f_score(GT, beta=betas[2]):.3f}{t(3)}{l_VM_pro_half.weighted_precision(GT, alpha):.3f}{t(3)}{l_VM_pro_half.recall(GT):.3f}
--------------------------------------------------------------------------
--------------------------------------------------------------------------

""")



if __name__ == '__main__':
	print(number_real_vids)
	print(number_fake_vids)
	save_logloss()
	save_mean_var()
	save_acc()
	save_confusion_matrix()
	save_f_scores((1, 2, 0.2), alpha=250)
	summary.close()
	st.plot_roc((sef, sef_pro, ntech, ntech_pro, VM, VM_pro), GT)
	st.plot_pr((sef, sef_pro, ntech, ntech_pro, VM, VM_pro), GT, alpha=10, thr_step=0.005)