import numpy as np
import h5py
from runs.eval_utils import save_confusion_matrix, predictive_entropy, mutual_info
import os

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x, axis=1, keepdims=True)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'

    from keras import Model
    from models import backbone
    from paths import submission_dir
    # from datasets.ISIC2018 import load_validation_data, load_test_data
    from misc_utils.prediction_utils import cyclic_stacking

    def task3_tta_predict(model, img_arr):
        img_arr_tta = cyclic_stacking(img_arr)
        pred_logits = np.zeros(shape=(img_arr.shape[0], 7))

        for _img_crops in img_arr_tta:
            pred_logits += model.predict(_img_crops)

        pred_logits = pred_logits/len(img_arr_tta)

        return pred_logits

    MC = True
    label_name = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
    backbone_name = 'densenet169'

    num_cls = 7
    # num_folds = 1
    dropout_rate = 0.5
    num_dense_layers = 1
    num_dense_units = 128
    pooling = 'avg'
    dense_layer_regularizer = 'L2'

    def one_hot(y):
        y_ohe = np.zeros((y.size, int(y.max() + 1)))
        y_ohe[np.arange(y.size), y.astype(int)] = 1
        return y_ohe

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'preprocessed_data.h5')
    h5f = h5py.File(data_dir, 'r')
    x_train = h5f['X_train'][:]
    y_train = one_hot(h5f['y_train'][:])
    x_valid = h5f['X_test'][:]
    y_valid = one_hot(h5f['y_test'][:])
    h5f.close()

    model_name = 'lesion_densenet169'
    run_name = model_name + '_2'
    root_path = os.path.dirname(os.path.abspath(__file__))
    dest_path = os.path.join(root_path, run_name)

    model = backbone(backbone_name).classification_model(load_weights_from=model_name,
                                                         num_classes=num_cls,
                                                         num_dense_layers=num_dense_layers,
                                                         num_dense_units=num_dense_units,
                                                         pooling=pooling,
                                                         dropout_rate=dropout_rate,
                                                         kernel_regularizer=dense_layer_regularizer)
    best_acc = 0.0
    best_mc_iter = 0
    for monte_carlo_simulations in range(1, 50, 1):
        y_prob = np.zeros((monte_carlo_simulations, y_valid.shape[0], num_cls))
        print('-'*50)
        print('-----------------------------RUNNING FOR MC={}-----------------------------------'.format(monte_carlo_simulations))
        for mc_iter in range(monte_carlo_simulations):
            print('running iter#{}'.format(mc_iter))
            y_prob[mc_iter] = model.predict(x_valid)

        mean_prob = y_prob.mean(axis=0)
        var_pred_entropy = predictive_entropy(mean_prob)
        var_pred_MI = mutual_info(y_prob)
        acc = 1 - np.count_nonzero(np.not_equal(mean_prob.argmax(axis=1), y_valid.argmax(axis=1))) / mean_prob.shape[0]

        if acc >= best_acc:
            best_acc = acc
            best_mc_iter = monte_carlo_simulations
            print('------------------------ MC = {0}, accuracy={1:.02%} (improved)'.format(monte_carlo_simulations, acc))
        else:
            print('------------------------ MC = {0}, accuracy={1:.02%}'.format(monte_carlo_simulations, acc))

        fig_path = os.path.join(dest_path, 'confusion_matrix_MC=' + str(monte_carlo_simulations) + '.png')
        save_confusion_matrix(y_valid.argmax(axis=1).astype(int), mean_prob.argmax(axis=1).astype(int),
                              classes=np.array(label_name),
                              dest_path=fig_path,
                              title='Confusion matrix, without normalization')

        fig_path = os.path.join(dest_path,
                                'normalized_confusion_matrix_MC=' + str(monte_carlo_simulations) + '.png')
        save_confusion_matrix(y_valid.argmax(axis=1).astype(int), mean_prob.argmax(axis=1).astype(int),
                              classes=np.array(label_name), normalize=True,
                              dest_path=fig_path,
                              title='Normalized confusion matrix')

        file_name = os.path.join(dest_path, 'normalized_MC=' + str(monte_carlo_simulations) + '.h5')
        h5f = h5py.File(file_name, 'w')
        h5f.create_dataset('y_prob', data=y_prob)
        h5f.create_dataset('mean_prob', data=mean_prob)
        h5f.create_dataset('var_pred_entropy', data=var_pred_entropy)
        h5f.create_dataset('var_pred_MI', data=var_pred_MI)
        h5f.close()

    print('------------------------ Best accuracy={0:.02%} achieved with MC={1}'.format(best_acc, best_mc_iter))
