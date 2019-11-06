import numpy as np
import matplotlib.pyplot as plt


def one_hot(y):
    y_ohe = np.zeros((y.size, int(y.max() + 1)))
    y_ohe[np.arange(y.size), y.astype(int)] = 1
    return y_ohe


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x, axis=1, keepdims=True)


def mutual_info(mc_prob):
    """
    computes the mutual information
    :param mc_prob: List MC probabilities of length mc_simulations;
                    each of shape  of shape [batch_size, num_cls]
    :return: mutual information of shape [batch_size, num_cls]
    """
    eps = 1e-5
    mean_prob = mc_prob.mean(axis=0)
    first_term = -1 * np.sum(mean_prob * np.log(mean_prob + eps), axis=1)
    second_term = np.sum(np.mean([prob * np.log(prob + eps) for prob in mc_prob], axis=0), axis=1)
    return first_term + second_term


def uncertainty_fraction_removal(y, y_pred, y_var, num_fracs, num_random_reps):
    fractions = np.linspace(1 / num_fracs, 1, num_fracs)
    num_samples = y.shape[0]
    acc_unc_sort = np.array([])
    acc_pred_sort = np.array([])
    acc_random_frac = np.zeros((0, num_fracs))

    remain_samples = []
    # uncertainty-based removal
    inds = y_var.argsort()
    y_sorted = y[inds]
    y_pred_sorted = y_pred[inds]
    for frac in fractions:
        y_temp = y_sorted[:int(num_samples * frac)]
        remain_samples.append(len(y_temp))
        y_pred_temp = y_pred_sorted[:int(num_samples * frac)]
        acc_unc_sort = np.append(acc_unc_sort, np.sum(y_temp == y_pred_temp) / y_temp.shape[0])

    # random removal
    for rep in range(num_random_reps):
        acc_random_sort = np.array([])
        perm = np.random.permutation(y_var.shape[0])
        y_sorted = y[perm]
        y_pred_sorted = y_pred[perm]
        for frac in fractions:
            y_temp = y_sorted[:int(num_samples * frac)]
            y_pred_temp = y_pred_sorted[:int(num_samples * frac)]
            acc_random_sort = np.append(acc_random_sort, np.sum(y_temp == y_pred_temp) / y_temp.shape[0])
        acc_random_frac = np.concatenate((acc_random_frac, np.reshape(acc_random_sort, [1, -1])), axis=0)
    acc_random_m = np.mean(acc_random_frac, axis=0)
    acc_random_s = np.std(acc_random_frac, axis=0)

    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(fractions, acc_unc_sort, 'o-', lw=1.5, label='uncertainty-based', markersize=3, color='royalblue')

    line1, = ax.plot(fractions, acc_random_m, 'o', lw=1, label='Random', markersize=3, color='black')
    ax.fill_between(fractions,
                    acc_random_m - acc_random_s,
                    acc_random_m + acc_random_s,
                    color='black', alpha=0.3)
    line1.set_dashes([1, 1, 1, 1])  # 2pt line, 2pt break, 10pt line, 2pt break

    ax.set_xlabel('Fraction of Retained Data')
    ax.set_ylabel('Prediction Accuracy')


def normalized_uncertainty_toleration_removal(y, y_pred, y_var, num_points):
    acc_uncertainty, acc_overall = np.array([]), np.array([])
    num_cls = len(np.unique(y))
    y_var = (y_var - y_var.min()) / (y_var.max() - y_var.min())
    per_class_remain_count = np.zeros((num_points, num_cls))
    per_class_acc = np.zeros((num_points, num_cls))
    thresholds = np.linspace(0, 1, num_points)
    remain_samples = []
    for i, t in enumerate(thresholds):
        idx = np.argwhere(y_var >= t)
        y_temp = np.delete(y, idx)
        remain_samples.append(len(y_temp))
        y_pred_temp = np.delete(y_pred, idx)
        acc_uncertainty = np.append(acc_uncertainty, np.sum(y_temp == y_pred_temp) / y_temp.shape[0])
        if len(y_temp):
            per_class_remain_count[i, :] = np.array([len(y_temp[y_temp == c]) for c in range(num_cls)])
            per_class_acc[i, :] = np.array(
                [np.sum(y_temp[y_temp == c] == y_pred_temp[y_temp == c]) / y_temp[y_temp == c].shape[0] for c in
                 range(num_cls)])

    plt.figure()
    plt.plot(thresholds, acc_uncertainty, lw=1.5, color='royalblue', marker='o', markersize=4)
    plt.xlabel('Normalized Tolerated Model Uncertainty')
    plt.ylabel('Prediction Accuracy')

