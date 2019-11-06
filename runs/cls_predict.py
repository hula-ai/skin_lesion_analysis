import numpy as np
import h5py
from runs.eval_utils import save_confusion_matrix, predictive_entropy, mutual_info
import os


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x, axis=1, keepdims=True)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'

    from keras import Model
    from models import backbone
    from misc_utils.prediction_utils import cyclic_stacking


    MC = False
    monte_carlo_simulations = 23
    label_name = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
    # backbone_name = 'my_densenet'
    backbone_name = 'densenet169'
    # backbone_name = 'vgg16'
    # backbone_name = 'resnet50'
    # backbone_name = 'densenet201'

    num_cls = 7
    # num_folds = 1
    dropout_rate = 0.5
    num_dense_layers = 1
    num_dense_units = 128
    pooling = 'avg'
    dense_layer_regularizer = 'L2'
    model_name = 'lesion_densenet169'


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

    run_name = model_name + '_2'
    root_path = os.path.dirname(os.path.abspath(__file__))
    dest_path = os.path.join(root_path, run_name)
    if not os.path.exists(dest_path):
        os.mkdir(dest_path)

    if not MC:  # normal evaluation
        model = backbone(backbone_name).classification_model(load_weights_from=model_name,
                                                             num_classes=num_cls,
                                                             num_dense_layers=num_dense_layers,
                                                             num_dense_units=num_dense_units,
                                                             dropout_rate=0.,
                                                             pooling=pooling,
                                                             kernel_regularizer=dense_layer_regularizer)
        y_pred = model.predict(x_valid)
        y_prob = softmax(y_pred)

        fig_path = os.path.join(dest_path, 'confusion_matrix' + '.png')
        save_confusion_matrix(y_valid.argmax(axis=1).astype(int), y_prob.argmax(axis=1).astype(int),
                              classes=np.array(label_name),
                              dest_path=fig_path,
                              title='Confusion matrix, without normalization')

        fig_path = os.path.join(dest_path, 'normalized_confusion_matrix' + '.png')
        save_confusion_matrix(y_valid.argmax(axis=1).astype(int), y_prob.argmax(axis=1).astype(int),
                              classes=np.array(label_name), normalize=True,
                              dest_path=fig_path,
                              title='Normalized confusion matrix')

        acc = 1 - np.count_nonzero(np.not_equal(y_prob.argmax(axis=1), y_valid.argmax(axis=1))) / y_prob.shape[0]
        print('accuracy={0:.02%}'.format(acc))

        file_name = os.path.join(base_dir, 'predictions.h5')
        h5f = h5py.File(file_name, 'w')
        h5f.create_dataset('x_valid', data=x_valid)
        h5f.create_dataset('y_valid', data=y_valid)
        h5f.create_dataset('y_prob', data=y_prob)
        h5f.close()

    else:       # MC-evaluation
        model = backbone(backbone_name).classification_model(load_weights_from=model_name,
                                                             num_classes=num_cls,
                                                             num_dense_layers=num_dense_layers,
                                                             num_dense_units=num_dense_units,
                                                             pooling=pooling,
                                                             dropout_rate=dropout_rate,
                                                             kernel_regularizer=dense_layer_regularizer,
                                                             print_model_summary=True)

        y_prob = np.zeros((monte_carlo_simulations, y_valid.shape[0], num_cls))
        for mc_iter in range(monte_carlo_simulations):
            print('running iter#{}'.format(mc_iter))
            y_prob[mc_iter] = model.predict(x_valid)

        mean_prob = y_prob.mean(axis=0)
        var_pred_entropy = predictive_entropy(mean_prob)
        var_pred_MI = mutual_info(y_prob)
        acc = 1 - np.count_nonzero(np.not_equal(mean_prob.argmax(axis=1), y_valid.argmax(axis=1))) / mean_prob.shape[0]
        print('accuracy={0:.02%}'.format(acc))

        fig_path = os.path.join(dest_path, 'confusion_matrix_MC=' + str(monte_carlo_simulations) + '.png')
        save_confusion_matrix(y_valid.argmax(axis=1).astype(int), mean_prob.argmax(axis=1).astype(int),
                              classes=np.array(label_name),
                              dest_path=fig_path,
                              title='Confusion matrix, without normalization')

        fig_path = os.path.join(dest_path, 'normalized_confusion_matrix_MC=' + str(monte_carlo_simulations) + '.png')
        save_confusion_matrix(y_valid.argmax(axis=1).astype(int), mean_prob.argmax(axis=1).astype(int),
                              classes=np.array(label_name), normalize=True,
                              dest_path=fig_path,
                              title='Normalized confusion matrix')

        file_name = os.path.join(base_dir, 'MC_predictions.h5')
        h5f = h5py.File(file_name, 'w')
        h5f.create_dataset('x_valid', data=x_valid)
        h5f.create_dataset('y_valid', data=y_valid)
        h5f.create_dataset('y_prob', data=y_prob)
        h5f.create_dataset('mean_prob', data=mean_prob)
        h5f.create_dataset('var_pred_entropy', data=var_pred_entropy)
        h5f.create_dataset('var_pred_MI', data=var_pred_MI)
        h5f.close()

