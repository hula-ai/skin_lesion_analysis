import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

from runs.eval_utils import save_confusion_matrix
from runs.utils_analysis import uncertainty_density_plot, class_based_density_plot, uncertainty_toleration_removal, \
    normalized_uncertainty_toleration_removal, uncertainty_fraction_removal, correlation_plot

MC = 10
num_intervals = 20
num_fractions = 20
reps_for_random = 40

#-------------------------- LOAD DATA------------------------------
run_name = '/home/cougarnet.uh.edu/amobiny/Desktop/ISIC2018/runs//DenseNet169_runs/balanced' \
           '_enddrop_weights_densenet169_1_normalized_MC=' + str(MC)
if not os.path.exists(run_name):
    os.makedirs(run_name)

if not os.path.exists(run_name + '/distribution'):
    os.makedirs(run_name + '/distribution')

if not os.path.exists(run_name + '/distribution_class_wise'):
    os.makedirs(run_name + '/distribution_class_wise')

if not os.path.exists(run_name + '/toleration_removal'):
    os.makedirs(run_name + '/toleration_removal')

if not os.path.exists(run_name + '/normalized_toleration_removal'):
    os.makedirs(run_name + '/normalized_toleration_removal')

if not os.path.exists(run_name + '/fraction_removal'):
    os.makedirs(run_name + '/fraction_removal')

if not os.path.exists(run_name + '/correlation_plot'):
    os.makedirs(run_name + '/correlation_plot')

data_dir = '/home/cougarnet.uh.edu/amobiny/Desktop/BNN_Skin_Lesion/data/ISIC/preprocessed_data.h5'
h5f = h5py.File(data_dir, 'r')
y = h5f['y_test'][:]
y_train = h5f['y_train'][:]
h5f.close()

file_path = run_name + '.h5'
h5f = h5py.File(file_path, 'r')
y_pred = np.argmax(h5f['mean_prob'][:], axis=1)
y_prob = h5f['y_prob'][:]
y_entropy = h5f['var_pred_entropy'][:]
y_MI = h5f['var_pred_MI'][:]
h5f.close()


y_entropy = (y_entropy - y_entropy.min()) / (y_entropy.max() - y_entropy.min())

# label_name = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
# save_confusion_matrix(y.astype(int), y_pred.astype(int),
#                       classes=np.array(label_name), normalize=True,
#                       dest_path=run_name + '/confusion.pdf',
#                       title='Normalized confusion matrix')
# save_confusion_matrix(y.astype(int), y_pred.astype(int),
#                       classes=np.array(label_name), normalize=True,
#                       dest_path=run_name + '/confusion.svg',
#                       title='Normalized confusion matrix')
# computing std as a measure of uncertainty
y_std_all_classes = np.std(y_prob, axis=0)
y_std_predicted_class = np.zeros(y.shape[0])  # if to use the standard deviation of the target class
for sample in range(y.shape[0]):
    y_std_predicted_class[sample] = y_std_all_classes[sample, y[sample].astype(int)]

y_std_mean = np.mean(y_std_all_classes, axis=-1)  # if to use the average over all classes

y_pred_prob = np.max(y_prob[5], axis=-1)


#-------------------------- DISTRIBUTION PLOT------------------------------
# uncertainty_density_plot(y, y_pred, y_entropy, run_name + '/distribution/entropy_all_classes')
# uncertainty_density_plot(y, y_pred, y_MI, run_name + '/distribution/MI_all_classes')
# uncertainty_density_plot(y, y_pred, y_std_mean, run_name + '/distribution/stdMean_all_classes')
# uncertainty_density_plot(y, y_pred, y_std_predicted_class, run_name + '/distribution/stdPred_all_classes')

#-------------------------- CLASS DISTRIBUTION PLOT------------------------
# class_based_density_plot(y, y_pred, y_entropy, run_name + '/distribution_class_wise/entropy_class_')
# class_based_density_plot(y, y_pred, y_MI, run_name + '/distribution_class_wise/MI_class_')
# class_based_density_plot(y, y_pred, y_std_mean, run_name + '/distribution_class_wise/stdMean_class_')
# class_based_density_plot(y, y_pred, y_std_predicted_class, run_name + '/distribution_class_wise/stdPred_class_')


#-------------------------- TOLERATION REMOVAL------------------------------
# uncertainty_toleration_removal(y, y_pred, y_entropy, num_intervals, run_name + '/toleration_removal/entropy_toleration_removal')
# uncertainty_toleration_removal(y, y_pred, y_MI, num_intervals, run_name + '/toleration_removal/MI_toleration_removal')
# uncertainty_toleration_removal(y, y_pred, y_std_mean, num_intervals, run_name + '/toleration_removal/stdMean_toleration_removal')
# uncertainty_toleration_removal(y, y_pred, y_std_predicted_class, num_intervals, run_name + '/toleration_removal/stdPred_toleration_removal')


#-------------------------- NORMALIZED TOLERATION REMOVAL------------------------------
normalized_uncertainty_toleration_removal(y, y_pred, y_entropy, y_pred_prob, num_intervals, run_name + '/normalized_toleration_removal/entropy_toleration_removal')
# normalized_uncertainty_toleration_removal(y, y_pred, y_MI, y_pred_prob, num_intervals, run_name + '/normalized_toleration_removal/MI_toleration_removal')
# normalized_uncertainty_toleration_removal(y, y_pred, y_std_mean, y_pred_prob, num_intervals, run_name + '/normalized_toleration_removal/stdMean_toleration_removal')


#-------------------------- FRACTION REMOVAL------------------------------
uncertainty_fraction_removal(y, y_pred, y_entropy, y_pred_prob, num_fractions, reps_for_random, run_name + '/fraction_removal/entropy_fraction')
# uncertainty_fraction_removal(y, y_pred, y_MI, y_pred_prob, num_fractions, reps_for_random, run_name + '/fraction_removal/MI_fraction')
# uncertainty_fraction_removal(y, y_pred, y_std_mean, y_pred_prob, num_fractions, reps_for_random, run_name + '/fraction_removal/stdMean_fraction')


#-------------------------- CORRELATION PLOT------------------------------
class_mean_entropy = np.array([np.mean(y_entropy[y == c]) for c in range(7)])
class_std_entropy = np.array([np.std(y_entropy[y == c]) for c in range(7)])

class_mean_MI = np.array([np.mean(y_MI[y == c]) for c in range(7)])
class_std_MI = np.array([np.std(y_MI[y == c]) for c in range(7)])

class_mean_std = np.array([np.mean(y_std_mean[y == c]) for c in range(7)])
class_std_std = np.array([np.std(y_std_mean[y == c]) for c in range(7)])

class_accuracy = np.array([np.sum(y[y == c] == y_pred[y == c]) / np.sum(y == c) for c in range(7)])
class_size = np.array([np.sum(y_train == c) for c in range(7)])

correlation_plot(class_mean_entropy, class_accuracy, class_size, run_name + '/correlation_plot/entropy_correlation')
# correlation_plot(class_mean_MI, class_accuracy, class_size, run_name + '/correlation_plot/MI_correlation')
# correlation_plot(class_mean_std, class_accuracy, class_size, run_name + '/correlation_plot/std_correlation')


# fig, axes = plt.subplots(nrows=1, ncols=2)
# plt.subplots_adjust(left=0.1, bottom=0.15, right=0.9, top=0.9, wspace=0.1, hspace=None)
#
#
# ax1 = axes[1]
# ax1.plot(all_thresholds, acc_do_frac, 'o-', lw=1.5, label='MC-Dropout', markersize=3, color='red')
# line1, = ax1.plot(all_thresholds, acc_random_m, 'o', lw=1, label='Random', markersize=2, color='black')
# ax1.fill_between(all_thresholds,
#                  acc_random_m - acc_random_s,
#                  acc_random_m + acc_random_s,
#                  color='black', alpha=0.3)
# line1.set_dashes([1, 1, 1, 2])  # 2pt line, 2pt break, 10pt line, 2pt break
#
# ax1.set_ylim([0.88, 1.])
# ax1.legend(loc='upper center', ncol=3)
#
# width = 7
# height = width / 2
# fig.set_size_inches(width, height)
# plt.show()
# fig.savefig('acc_frac.pdf')

print()
