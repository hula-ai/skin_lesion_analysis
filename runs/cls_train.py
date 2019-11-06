import numpy as np
import os
from models import backbone
from callback import config_cls_callbacks
from misc_utils.eval_utils import compute_class_weights
from misc_utils.print_utils import log_variable, Tee
from misc_utils.filename_utils import get_log_filename
from misc_utils.visualization_utils import BatchVisualization
from keras.preprocessing.image import ImageDataGenerator
import sys
import h5py

os.environ['CUDA_VISIBLE_DEVICES'] = '2'


if __name__ == '__main__':
    # backbone_name = 'vgg16'
    #backbone_name = 'resnet50'
    backbone_name = 'densenet169'
    # backbone_name = 'densenet201'
    # backbone_name = 'my_densenet'
    # backbone_name = 'inception_v3'

    # Network architecture related params
    dropout_rate = 0.5
    backbone_options = {}
    num_dense_layers = 1
    num_dense_units = 128
    pooling = 'avg'

    # Training related params
    dense_layer_regularizer = 'L2'
    class_wt_type = 'balanced'              # 'ones' or 'balanced' or 'balanced-sqrt'
    lr = 1e-4

    # num_folds = 5

    # for k_fold in range(num_folds):

    version = '_enddrop_weights_'

    # run_name = 'task3_' + backbone_name + '_k' + str(k_fold) + '_v' + version

    run_name = 'task3_' + backbone_name + '_vfinal' + version + class_wt_type

    # Set prev_run_name to continue training from a previous run
    prev_run_name = None

    logfile = open(get_log_filename(run_name=run_name), 'w+')
    original = sys.stdout
    sys.stdout = Tee(sys.stdout, logfile)

    # (x_train, y_train), (x_valid, y_valid), _ = load_training_data(task_idx=3,
    #                                                                output_size=224,
    #                                                                num_partitions=num_folds,
    #                                                                idx_partition=k_fold)

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

    num_classes = y_train.shape[1]

    callbacks = config_cls_callbacks(run_name)

    model = backbone(backbone_name, **backbone_options).classification_model(
        input_shape=x_train.shape[1:],
        num_classes=num_classes,
        num_dense_layers=num_dense_layers,
        num_dense_units=num_dense_units,
        pooling=pooling,
        dropout_rate=dropout_rate,
        kernel_regularizer=dense_layer_regularizer,
        save_to=run_name,
        load_from=prev_run_name,
        print_model_summary=True,
        plot_model_summary=False,
        lr=lr)

    n_samples_train = x_train.shape[0]
    n_samples_valid = x_valid.shape[0]

    class_weights = compute_class_weights(y_train, wt_type=class_wt_type)

    batch_size = 8
    use_data_aug = True
    horizontal_flip = True
    vertical_flip = True
    rotation_angle = 180
    width_shift_range = 0.1
    height_shift_range = 0.1

    log_variable(var_name='num_dense_layers', var_value=num_dense_layers)
    log_variable(var_name='num_dense_units', var_value=num_dense_units)
    log_variable(var_name='dropout_rate', var_value=dropout_rate)
    log_variable(var_name='pooling', var_value=pooling)
    log_variable(var_name='class_wt_type', var_value=class_wt_type)
    log_variable(var_name='dense_layer_regularizer', var_value=dense_layer_regularizer)
    log_variable(var_name='class_wt_type', var_value=class_wt_type)
    log_variable(var_name='learning_rate', var_value=lr)
    log_variable(var_name='batch_size', var_value=batch_size)

    log_variable(var_name='use_data_aug', var_value=use_data_aug)

    if use_data_aug:

        log_variable(var_name='horizontal_flip', var_value=horizontal_flip)
        log_variable(var_name='vertical_flip', var_value=vertical_flip)
        log_variable(var_name='width_shift_range', var_value=width_shift_range)
        log_variable(var_name='height_shift_range', var_value=height_shift_range)
        log_variable(var_name='rotation_angle', var_value=rotation_angle)

    log_variable(var_name='n_samples_train', var_value=n_samples_train)
    log_variable(var_name='n_samples_valid', var_value=n_samples_valid)

    sys.stdout.flush()  # need to make sure everything gets written to file

    if use_data_aug:

        datagen = ImageDataGenerator(rotation_range=rotation_angle,
                                     horizontal_flip=horizontal_flip,
                                     vertical_flip=vertical_flip,
                                     width_shift_range=width_shift_range,
                                     height_shift_range=height_shift_range)

        model.fit_generator(generator=datagen.flow(x_train, y_train, batch_size=batch_size),
                            steps_per_epoch=x_train.shape[0] // batch_size,
                            epochs=1000,
                            initial_epoch=0,
                            verbose=1,
                            validation_data=(x_valid, y_valid),
                            callbacks=callbacks,
                            workers=8,
                            use_multiprocessing=True)

    else:

        model.fit(x=x_train,
                  y=y_train,
                  batch_size=batch_size,
                  epochs=1000,
                  verbose=1,
                  validation_data=(x_valid, y_valid),
                  class_weight=class_weights,
                  shuffle=True,
                  callbacks=callbacks)

    sys.stdout = original

