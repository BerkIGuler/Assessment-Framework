import numpy as np
import sklearn.metrics
import matplotlib.pyplot as plt
import copy

class statistics():
	

	def __init__(self, file_name=None, name=None):

		self.data = {}
		self.name = name

		if file_name:
			with open(file_name, 'r') as fin:
				#skip the first line
				next(fin)
				for line in fin:
					line = line.split(',')
					self.data[line[0]] = float(line[1])


	def into_matrix(self):

		sorted_lst = sorted(list(self.data.items()), key=lambda x: x[0])
		sorted_arr = np.array(sorted_lst)

		return sorted_arr



	def get_score(self, vid_name):	
		return self.data[vid_name]



	def get_logloss(self, GT_labels):

		if len(self.data) != len(GT_labels.data):
			raise ValueError('Dimensions do not match!!!') 

		total_logloss = 0

		N = len(self.data)

		for vid in self.data:
			y = GT_labels.data[vid]
			y_pred = self.data[vid]
			ith_logloss = statistics.log_loss(y, y_pred)
			total_logloss += ith_logloss

		return - total_logloss / N



	def assign_labels(self, threshold, inplace=True):


		if inplace:

			for key in self.data:
				if self.data[key] > threshold:
					self.data[key] = 1
				else:
					self.data[key] = 0

		else:

			new_model = statistics()

			for key in self.data:
				if self.data[key] > threshold:
					new_model.data[key] = 1
				else:
					new_model.data[key] = 0

			return new_model





	def confusion_matrix(self, GT):

		#true positive
		detected_true = 0

		for vid in GT.data:
			if GT.data[vid] == 1:
				if self.data[vid] == 1:
					detected_true += 1

		tp = detected_true 

		#false positive 
		detected_true = 0

		for vid in GT.data:
			if GT.data[vid] == 0:
				if self.data[vid] == 1:
					detected_true += 1

		fp = detected_true 

		#true negative 
		detected_false = 0

		for vid in GT.data:
			if GT.data[vid] == 0:
				if self.data[vid] == 0:
					detected_false += 1

		tn = detected_false 

		#false negative rate
		detected_false = 0

		for vid in GT.data:
			if GT.data[vid] == 1:
				if self.data[vid] == 0:
					detected_false += 1

		fn = detected_false 

		return np.array([[tn, fp],
						 [fn, tp]])





	def fn_rate(self, GT):

		confusion_m = self.confusion_matrix(GT)
		tn, fp, fn, tp = confusion_m.ravel()

		return fn / (fn + tp)

	def fp_rate(self, GT):

		confusion_m = self.confusion_matrix(GT)
		tn, fp, fn, tp = confusion_m.ravel()

		return fp / (fp + tn)


	def tn_rate(self, GT):

		confusion_m = self.confusion_matrix(GT)
		tn, fp, fn, tp = confusion_m.ravel()

		return tn / (tn + fp)

	def tp_rate(self, GT):

		confusion_m = self.confusion_matrix(GT)
		tn, fp, fn, tp = confusion_m.ravel()

		return tp / (tp + fn)


	def acc(self, GT):
		confusion_m = self.confusion_matrix(GT)
		tn, fp, fn, tp = confusion_m.ravel()

		return (tp + tn) / (tp + tn + fn + fp)

	@staticmethod		
	def log_loss(y, y_pred):
		return y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)


	def plot_roc(models, GT):

		COLORS = ('b', 'g', 'r', 'c', 'm', 'y', 'k', 'w')
		LINESTY = (None, '--')


		plt.figure(figsize=(12, 12))
		plt.plot([0, 1], [0, 1], color="k", linestyle="--")
		plt.xlim([0.0, 1.0])
		plt.ylim([0.0, 1.05])
		plt.xlabel("False Positive Rate", fontsize=18)
		plt.ylabel("True Positive Rate", fontsize=18)
		plt.title("Receiver Operating Characteristic Curve", fontsize=25)
		plt.legend(loc="lower right")

		for i, model in enumerate(models):
			tpr, fpr =  model.roc_scores(GT, thr_step=0.01)
			roc_auc = sklearn.metrics.auc(fpr, tpr)
			plt.plot(fpr, tpr, color=COLORS[i//2], linestyle=LINESTY[i%2], label=f"{model.name}, AUC = {roc_auc.round(4)}")

		plt.legend(loc="lower right", fontsize=16)
		plt.savefig('ROC.png')


	def plot_pr(models, GT, alpha=100, thr_step=0.01):

		COLORS = ('b', 'g', 'r', 'c', 'm', 'y', 'k', 'w')
		LINESTY = (None, '--')


		plt.figure(figsize=(12, 12))
		plt.xlim([0.0, 1.0])
		plt.ylim([0.0, 1.05])
		plt.xlabel("Recall", fontsize=18)
		plt.ylabel("Weighted Precision", fontsize=18)
		plt.title(f"Weighted Precision vs. Recall (alpha = {alpha})", fontsize=25)
		plt.legend(loc="lower right")

		for i, model in enumerate(models):
			wp, rec =  model.pr_scores(GT, thr_step, alpha)
			pr_auc = sklearn.metrics.auc(rec, wp)
			plt.plot(rec, wp, color=COLORS[i//2], linestyle=LINESTY[i%2], label=f"{model.name}")

		plt.legend(loc="lower left", fontsize=16)
		plt.savefig('PR.png')

	def mean(self):
		return np.mean(self.into_matrix()[:, 1].astype(np.float32))

	def var(self):
		return np.var(self.into_matrix()[:, 1].astype(np.float32))

	def find_best_threshold(self, GT):

		best_accuracy = 0

		thr = 0.0
		best_thr = None

		while thr < 0.95:	


			new_model = self.assign_labels(thr, inplace=False)
			acc = new_model.acc(GT)

			if acc > best_accuracy:
				best_accuracy = acc
				best_thr = thr


			thr += 0.005

		return best_thr


	def weighted_precision(self, GT, alpha):

		confusion_m = self.confusion_matrix(GT)
		tn, fp, fn, tp = confusion_m.ravel()

		return (tp) / (tp + alpha * fp)


	def recall(self, GT):

		confusion_m = self.confusion_matrix(GT)
		tn, fp, fn, tp = confusion_m.ravel()

		return (tp) / (tp + fn)

	def roc_scores(self, GT, thr_step=0.01):

		thrs = np.arange(0, 1, thr_step)

		tpr = np.zeros((thrs.shape[0]))
		fpr = np.zeros((thrs.shape[0]))

		for i in range(thrs.shape[0]):
			new_model = self.assign_labels(thrs[i], inplace=False)
			fpr[i] = new_model.fp_rate(GT)
			tpr[i] = new_model.tp_rate(GT)


		return tpr, fpr


	def pr_scores(self,  GT, thr_step=0.01, alpha=100):

		thrs = np.arange(0, 1, thr_step)

		wp = np.zeros((thrs.shape[0]))
		rec = np.zeros((thrs.shape[0]))

		for i in range(thrs.shape[0]):
			new_model = self.assign_labels(thrs[i], inplace=False)
			wp[i] = new_model.weighted_precision(GT, alpha)
			rec[i] = new_model.recall(GT)


		return wp, rec



	def f_score(self, GT, beta=1):
		confusion_m = self.confusion_matrix(GT)
		tn, fp, fn, tp = confusion_m.ravel()

		return ((1 + beta**2) * tp) / ((1 + beta**2)*tp + fn*beta**2 + fp)









		








