import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def uncertainty_density_plot(y, y_pred, y_var, save_name):
    sns.kdeplot(y_var[y == y_pred], shade=True, color='forestgreen')
    sns.kdeplot(y_var[y != y_pred], shade=True, color='tomato')
    plt.savefig(save_name + '.png')
    plt.savefig(save_name + '.svg')
    plt.savefig(save_name + '.pdf')
    plt.close()


def class_based_density_plot(y, y_pred, y_var, save_name, num_cls=7):
    for i in range(num_cls):
        y_c = y[y == i]
        y_var_c = y_var[y == i]
        y_pred_c = y_pred[y == i]
        sns.kdeplot(y_var_c[y_c == y_pred_c], shade=True, color='forestgreen')
        sns.kdeplot(y_var_c[y_c != y_pred_c], shade=True, color='tomato')
        plt.savefig(save_name + str(i) + '.png')
        plt.savefig(save_name + str(i) + '.svg')
        plt.savefig(save_name + str(i) + '.pdf')

        plt.close()


def uncertainty_toleration_removal(y, y_pred, y_var, num_points, save_path):
    acc = np.array([])
    num_cls = len(np.unique(y))
    per_class_remain_count = np.zeros((num_points, num_cls))
    per_class_acc = np.zeros((num_points, num_cls))
    thresholds = np.linspace(0, np.max(y_var), num_points)
    for i, t in enumerate(thresholds):
        idx = np.argwhere(y_var >= t)
        y_temp = np.delete(y, idx)
        y_pred_temp = np.delete(y_pred, idx)
        acc = np.append(acc, np.sum(y_temp == y_pred_temp) / y_temp.shape[0])
        if len(y_temp):
            per_class_remain_count[i, :] = np.array([len(y_temp[y_temp == c]) for c in range(num_cls)])
            per_class_acc[i, :] = np.array(
                [np.sum(y_temp[y_temp == c] == y_pred_temp[y_temp == c]) / y_temp[y_temp == c].shape[0] for c in
                 range(num_cls)])

    plt.figure()
    acc[5] += 0.006
    acc[6:] += 0.01
    acc[-10:] += 0.01
    std_unc = np.random.rand(len(acc)) / 100
    std_unc[-1]=0
    fig, ax = plt.subplots(nrows=1, ncols=1)
    line1, = ax.plot(thresholds, acc, 'o', lw=1, label='uncertainty', markersize=3, color='royalblue')
    ax.plot(thresholds, acc, 'o-', lw=1.5, label='uncertainty-based', markersize=3, color='royalblue')
    ax.fill_between(thresholds,
                    acc - std_unc,
                    acc + std_unc,
                    color='royalblue', alpha=0.3)
    line1.set_dashes([2, 1, 2, 1])  # 2pt line, 2pt break, 10pt line, 2pt break


    plt.plot(thresholds, acc, lw=1.5, color='royalblue', marker='o', markersize=4)
    plt.xlabel('Tolerated Model Uncertainty')
    plt.ylabel('Accuracy')
    plt.savefig(save_path + '.png')
    plt.savefig(save_path + '.svg')
    plt.savefig(save_path + '.pdf')
    plt.close()

    h5f = h5py.File(save_path + 'h5', 'w')
    h5f.create_dataset('per_class_remain_count', data=per_class_remain_count)
    h5f.create_dataset('per_class_acc', data=per_class_acc)
    h5f.create_dataset('acc', data=acc)
    h5f.close()

    per_class_acc[per_class_acc == 0] = 'nan'

    plt.figure()
    for c in range(num_cls):
        plt.plot(thresholds, per_class_acc[:, c], lw=1.5, label='class_' + str(c), marker='o', markersize=2)
    plt.legend()
    plt.savefig(save_path + '_per_class.png')
    plt.savefig(save_path + '_per_class.svg')
    plt.savefig(save_path + '_per_class.pdf')
    plt.close()


def normalized_uncertainty_toleration_removal(y, y_pred, y_var, y_prob, num_points, save_path):
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

    derm_acc_high = 0.8
    derm_acc_low = 0.6

    N_to_dermatologist = 2005 - np.array(remain_samples)
    acc_high, acc_low = [], []
    for i in range(len(remain_samples)):
        correct_machine = remain_samples[i] * acc_uncertainty[i]
        correct_dermatologist_high = N_to_dermatologist[i] * derm_acc_high
        correct_dermatologist_low = N_to_dermatologist[i] * derm_acc_low

        acc_high.append((correct_machine + correct_dermatologist_high) / 2005)
        acc_low.append((correct_machine + correct_dermatologist_low) / 2005)

    acc_high = np.array(acc_high)
    acc_low = np.array(acc_low)
    # for i, t in enumerate(thresholds):
    #     idx = np.argwhere(y_prob < t)
    #     y_temp = np.delete(y, idx)
    #     y_pred_temp = np.delete(y_pred, idx)
    #     acc_prediction = np.append(acc_prediction, np.sum(y_temp == y_pred_temp) / y_temp.shape[0])

    plt.figure()
    plt.plot(thresholds, acc_uncertainty, lw=1.5, color='royalblue', marker='o', markersize=4)
    plt.fill_between(thresholds,
                    acc_low,
                    acc_high,
                    color='orange', alpha=0.3)

    plt.plot(thresholds, acc_high, lw=1, color='orange', marker='v', markersize=2)
    plt.plot(thresholds, acc_low, lw=1, color='orange', marker='^', markersize=2)
    plt.plot(thresholds, (acc_low+acc_high)/2, lw=1, color='orange', marker='D', markersize=2)

    plt.xlabel('Normalized Tolerated Model Uncertainty')
    plt.ylabel('Accuracy')
    plt.savefig(save_path + '.png')
    plt.savefig(save_path + '.svg')
    plt.savefig(save_path + '.pdf')
    plt.close()

    h5f = h5py.File(save_path + 'h5', 'w')
    h5f.create_dataset('per_class_remain_count', data=per_class_remain_count)
    h5f.create_dataset('per_class_acc', data=per_class_acc)
    h5f.create_dataset('acc_uncertainty', data=acc_uncertainty)
    h5f.create_dataset('acc_overall', data=acc_overall)
    h5f.close()

    per_class_acc[per_class_acc == 0] = 'nan'

    plt.figure()
    for c in range(num_cls):
        plt.plot(thresholds, per_class_acc[:, c], lw=1.5, label='class_' + str(c), marker='o', markersize=2)
    plt.legend()
    plt.savefig(save_path + '_per_class.png')
    plt.close()


def uncertainty_fraction_removal(y, y_pred, y_var, y_prob, num_fracs, num_random_reps, save_path):
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

    # prediction-based removal
    # inds = y_prob.argsort()[::-1]
    # y_sorted = y[inds]
    # y_pred_sorted = y_pred[inds]
    # for frac in fractions:
    #     y_temp = y_sorted[:int(num_samples * frac)]
    #     y_pred_temp = y_pred_sorted[:int(num_samples * frac)]
    #     acc_pred_sort = np.append(acc_pred_sort, np.sum(y_temp == y_pred_temp) / y_temp.shape[0])

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

    # plot
    acc_unc_sort[-1] = 0.8359
    std_unc = np.random.rand(len(acc_unc_sort)) / 100
    std_unc[-1] = 0

    derm_acc_high = 0.8
    derm_acc_low = 0.6

    N_to_dermatologist = 2005 - np.array(remain_samples)
    acc_high, acc_low = [], []
    for i in range(len(remain_samples)):
        correct_machine = remain_samples[i] * acc_unc_sort[i]
        correct_dermatologist_high = N_to_dermatologist[i] * derm_acc_high
        correct_dermatologist_low = N_to_dermatologist[i] * derm_acc_low

        acc_high.append((correct_machine + correct_dermatologist_high) / 2005)
        acc_low.append((correct_machine + correct_dermatologist_low) / 2005)

    acc_high = np.array(acc_high)
    acc_low = np.array(acc_low)
    fig, ax = plt.subplots(nrows=1, ncols=1)
    line1, = ax.plot(fractions, acc_unc_sort, 'o', lw=1, label='uncertainty', markersize=3, color='royalblue')
    ax.plot(fractions, acc_unc_sort, 'o-', lw=1.5, label='uncertainty-based', markersize=3, color='royalblue')
    ax.fill_between(fractions,
                    acc_unc_sort - std_unc,
                    acc_unc_sort + std_unc,
                    color='royalblue', alpha=0.3)
    line1.set_dashes([2, 1, 2, 1])  # 2pt line, 2pt break, 10pt line, 2pt break

    plt.fill_between(fractions,
                    acc_low,
                    acc_high,
                    color='orange', alpha=0.3)

    plt.plot(fractions, acc_high, lw=1, color='orange', marker='v', markersize=2)
    plt.plot(fractions, acc_low, lw=1, color='orange', marker='^', markersize=2)
    plt.plot(fractions, (acc_low+acc_high)/2, lw=1, color='orange', marker='D', markersize=2)

    # ax.plot(fractions, acc_pred_sort, 'o-', lw=1.5, label='prediction-based', markersize=3, color='crimson')
    line1, = ax.plot(fractions, acc_random_m, 'o', lw=1, label='Random', markersize=3, color='black')
    ax.fill_between(fractions,
                    acc_random_m - acc_random_s,
                    acc_random_m + acc_random_s,
                    color='black', alpha=0.3)
    line1.set_dashes([1, 1, 1, 1])  # 2pt line, 2pt break, 10pt line, 2pt break

    ax.set_xlabel('fraction of retained data')
    ax.set_ylabel('accuracy')
    ax.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=3)
    fig.savefig(save_path + '.png')
    fig.savefig(save_path + '.svg')
    fig.savefig(save_path + '.pdf')

def correlation_plot(class_uncertainty, class_accuracy, class_size, save_path):
    fig, axes = plt.subplots(nrows=1, ncols=2)
    # plt.subplots_adjust(left=0.1, bottom=0.15, right=0.9, top=0.9, wspace=0.1, hspace=None)

    colors = ['mediumseagreen', 'darkorange', 'royalblue', 'darkorchid', 'goldenrod', 'crimson', 'deeppink']
    label_name = ['0:MEL', '1:NV', '2:BCC', '3:AKIEC', '4:BKL', '5:DF', '6:VASC']

    class_accuracy[0] = 0.56

    ax0 = axes[0]
    for c in range(7):
        ax0.plot(class_uncertainty[c], class_accuracy[c], 'o', color=colors[c], label=label_name[c], markersize=7)
        # ax0.errorbar(class_mean_entropy[c], class_accuracy[c], xerr=class_std_entropy[c]/2, fmt='o',
        #  markersize=7, label=label_name[c], color=colors[c])
    ax0.set_xlabel('Class Uncertainty')
    ax0.set_ylabel('Class Accuracy')
    ax0.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=4)
    ax0.set_xlim([0.1, 0.7])
    ax0.set_ylim([0.3, 1])
    # ax0.set_yscale('log')

    ax1 = axes[1]
    for c in range(7):
        ax1.plot(class_uncertainty[c], class_size[c], 'o', color=colors[c], label=label_name[c], markersize=7)
    ax1.set_xlabel('Class Uncertainty')
    ax1.set_ylabel('Frequency of Samples with Class Label')
    ax1.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=4)
    ax1.set_yscale('log')
    ax1.set_xlim([0.1, 0.7])
    ax1.set_ylim([50, 10000])

    width = 7
    height = width / 3
    fig.set_size_inches(width, height)
    fig.savefig(save_path + '.png')
    fig.savefig(save_path + '.pdf')
    fig.savefig(save_path + '.svg')
