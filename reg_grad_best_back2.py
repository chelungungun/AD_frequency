"""
"""
import os
import time
import logging
import argparse
from scipy.fft import dct, idct
from tensorflow.keras.optimizers import Adam, SGD
# from keras.optimizers import Adam, SGD
from termcolor import colored
from tensorflow.keras.utils import to_categorical
# from keras.utils import to_categorical
import sklearn.metrics as metrics
import math
from art.attacks.evasion import ProjectedGradientDescent
from art.classifiers import KerasClassifier
import numpy as np
import copy
import tensorflow.keras.backend as K
# import keras.backend as K
from module_models_mclass import  small_cnn_reg_grad, wrn_cifar10_reg_grad, wrn_cifar10_reg_grad_back2
from tensorflow.python.framework.ops import disable_eager_execution
import tensorflow as tf
disable_eager_execution()
tf.compat.v1.experimental.output_all_intermediates(True)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def cat_cross(ytrue, ypred):
    eps = 0.000000000001
    loss = np.mean(-np.sum(ytrue * np.log(ypred + eps), axis=-1))

    return loss

def distance(i, j, imageSize, r):
    dis = np.sqrt((i - 0) ** 2 + (j - 0) ** 2)
    if dis < r:
        return 1.0
    else:
        return 0

def mask_radial(img_shape, thresh):
    batch, rows, cols, chan = img_shape
    mask = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            mask[i, j] = distance(i, j, imageSize=rows, r=thresh)
    mask = np.tile(np.expand_dims(mask, axis=-1), (1,1,chan))
    return mask

def filter(data, thresh, mode):
    img_shape = np.shape(data)
    X1 = np.transpose(data, axes=(0, 3, 1, 2))
    X2 = dct(X1, type=2, norm='ortho')
    X2_t = np.transpose(X2, axes=[0, 1, 3, 2])
    X3 = dct(X2_t, type=2, norm='ortho')
    X3_t = np.transpose(X3, axes=[0, 3, 2, 1])

    mask = mask_radial(img_shape, thresh)
    if mode == 'low_pass':
        mask = mask
    else:
        mask = 1 - mask
    X3_t_mask = X3_t * mask

    X1 = np.transpose(X3_t_mask, axes=[0, 3, 1, 2])
    X2 = idct(X1, type=2, norm='ortho')
    X2_t = np.transpose(X2, axes=[0, 1, 3, 2])
    X3 = idct(X2_t, type=2, norm='ortho')
    X3_t = np.transpose(X3, axes=[0, 3, 2, 1])

    return X3_t

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=2020, type=int)
    parser.add_argument('--dataset', default="mnist", type=str, choices=['mnist', 'cifar10', 'cifar10T'])
    parser.add_argument('--model', default='CNN', type=str)
    parser.add_argument('--fil', default="no", type=str)
    parser.add_argument('--fil-thresh', default=0, type=float)
    parser.add_argument('--fil-mode', default="low_pass", type=str, choices=['low_pass', 'high_pass'])
    parser.add_argument('--train-method', default="reg_grad", type=str)
    parser.add_argument('--clean-train', default=1, type=int)
    parser.add_argument('--adv-train', default=0, type=int)
    parser.add_argument('--coe-time', default=0, type=float)
    parser.add_argument('--coe-dct', default=0, type=float)
    parser.add_argument('--reg-norm', default='l1', type=str, choices=['l1', 'l2'])
    parser.add_argument('--reg-loss', default='entropy', type=str, choices=['entropy', 'mse'])
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--optimizer', default='Adam', type=str, choices=['Adam', 'SGD'])
    parser.add_argument('--grad-layer', default='input', type=str, choices=['input', 'back2'])
    parser.add_argument('--diem-back2', default=512, type=int)
    parser.add_argument('--reg-mode-dct', default='normal', type=str, choices=['low_pass', 'high_pass', 'normal'])
    parser.add_argument('--alpha', default=0.5, type=float)
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--epoch', default=200, type=int)
    parser.add_argument('--eps-adtrain-mnist', default=0.1, type=float)
    parser.add_argument('--eps-step-adtrain-mnist', default=0.02, type=float)
    parser.add_argument('--max-iter-adtrain-mnist', default=10, type=int)
    parser.add_argument('--eps-adtrain', default=0.031, type=float)
    parser.add_argument('--eps-step-adtrain', default=0.007, type=float)
    parser.add_argument('--max-iter-adtrain', default=10, type=int)
    parser.add_argument('--eps-attack-mnist', default=0.3, type=float)
    parser.add_argument('--eps-step-attack-mnist', default=0.01, type=float)
    parser.add_argument('--max-iter-attack-mnist', default=40, type=int)
    parser.add_argument('--eps-attack', default=0.031, type=float)
    parser.add_argument('--eps-step-attack', default=0.007, type=float)
    parser.add_argument('--max-iter-attack', default=10, type=int)

    return parser.parse_args()

def main():
    disable_eager_execution()
    tf.compat.v1.experimental.output_all_intermediates(True)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    args = get_args()
    glo_seed = args.seed
    tf.random.set_seed(glo_seed)
    np.random.seed(glo_seed)
    ## Load data
    dataset = args.dataset
    # dataset = 'cifar10'
    # dataset = 'cifar10T'
    fil = args.fil
    fil_thresh = args.fil_thresh
    fil_mode = args.fil_mode

    if dataset == 'mnist' or dataset == 'cifar10' or dataset == 'imagenet':
        x_train = np.load("multi_label_dataset/" + dataset + '_trainx.npy')
        x_test = np.load("multi_label_dataset/" + dataset + '_testx.npy')
        y_train = np.load("multi_label_dataset/" + dataset + '_trainy.npy')
        y_test = np.load("multi_label_dataset/" + dataset + '_testy.npy')
        # preprocess
        if dataset == 'mnist' or dataset == 'cifar10':
            x_train = x_train.astype('float32')
            x_test = x_test.astype('float32')
            x_train = x_train / 255.0
            x_test = x_test / 255.0

        #build validation data, 1000samples
        num_of_sample = np.shape(x_train)[0]
        index_all = np.random.permutation(num_of_sample, )
        num_val = 1000
        x_val = copy.deepcopy(x_train[index_all[0:num_val]])
        y_val = copy.deepcopy(y_train[index_all[0:num_val]])
        index_train = index_all[num_val:]
    else:
        x_train = np.load("multi_label_dataset/" + dataset + '_trainx.npy')
        x_test = np.load("multi_label_dataset/" + dataset + '_testx.npy')
        x_val = np.load("multi_label_dataset/" + dataset + '_valx.npy')  # 1000 samples
        y_train = np.load("multi_label_dataset/" + dataset + '_trainy.npy')
        y_val = np.load("multi_label_dataset/" + dataset + '_valy.npy')
        y_test = np.load("multi_label_dataset/" + dataset + '_testy.npy')

        # build validation data, 1000samples
        num_of_sample = np.shape(x_train)[0]  # 49000
        index_train = np.random.permutation(num_of_sample, )

    if fil == 'yes':
        x_val = filter(copy.deepcopy(x_val), thresh=fil_thresh, mode=fil_mode)
        x_train = filter(copy.deepcopy(x_train), thresh=fil_thresh, mode=fil_mode)
        # x_test = filter(copy.deepcopy(x_test), thresh=fil_thresh, mode=fil_mode)

    ##Build model
    train_method = args.train_method
    clean_adv = [args.clean_train, args.adv_train] # To control whether to do adversarial training.
    reg_coe_time = args.coe_time
    reg_coe_dct = args.coe_dct
    reg_norm = args.reg_norm
    reg_loss = args.reg_loss
    lr = args.lr # 0.1 for sgd, 0.0001 for adam
    weight_decay=args.weight_decay
    opt = args.optimizer
    if opt == 'Adam':
        optimizer = Adam(learning_rate=lr)
    else:
        optimizer = SGD(learning_rate=lr, momentum=0.9, decay=weight_decay)

    ## build reg weight
    reg_mode_dct = args.reg_mode_dct
    alpha = args.alpha
    grad_layer = args.grad_layer
    diem_back2 = args.diem_back2
    if grad_layer == 'input':
        weights = np.ones(shape=(x_train.shape[1],x_train.shape[2]), dtype='float32')
        for i in range(x_train.shape[1]):
            for j in range(x_train.shape[2]):
                weights[i][j] = np.sqrt(i^2 + j^2)
        if reg_mode_dct == 'low_pass':
            weights = np.expand_dims(np.exp(weights * alpha), axis=-1)
        elif reg_mode_dct == 'high_pass':
            weights = np.expand_dims(np.exp(weights * alpha * -1), axis=-1)
        else:
            weights = np.expand_dims(np.exp(weights * 0), axis=-1)
        if dataset == 'mnist':
            weights = weights / np.sum(weights)
        else:
            weights = np.tile(weights / np.sum(weights), (1,1,3))
    else:
        weights = np.square(np.arange(diem_back2))
        if reg_mode_dct == 'low_pass':
            weights = weights
        elif reg_mode_dct == 'high_pass':
            weights = 1 / weights
        else:
            weights = np.ones(diem_back2)
        weights = weights / np.sum(weights)

    inputshape_x = (x_train.shape[1], x_train.shape[2], x_train.shape[3])
    inputshape_y = (y_train.shape[1],)
    if dataset == 'mnist':
        model_adv, model, fun_grad_time = small_cnn_reg_grad(inputshape_x=inputshape_x, inputshape_y=inputshape_y, reg_coe_time=reg_coe_time, reg_coe_dct = reg_coe_dct, reg_norm=reg_norm, reg_loss=reg_loss, weights=weights)
    else:
        model_adv, model, fun_grad_time = wrn_cifar10_reg_grad_back2(inputshape_x=inputshape_x, inputshape_y=inputshape_y, reg_coe_time=reg_coe_time, reg_coe_dct = reg_coe_dct, reg_norm=reg_norm, reg_loss=reg_loss, weights=weights)  ##get wrn-34-10

    print(model.summary())
    model.compile(optimizer=optimizer, loss="categorical_crossentropy",metrics=["acc"])

    ##Train
    batch_size = args.batch_size
    n_epochs = args.epoch
    val_acc_best = 0
    if dataset == 'mnist' or dataset == 'cifar10':
        num_train = num_of_sample - num_val
    else:
        num_train = num_of_sample
    if dataset == 'mnist':
        eps = args.eps_adtrain_mnist
        eps_step = args.eps_step_adtrain_mnist
        max_iter_adtrain = args.max_iter_adtrain_mnist
    else:
        eps = args.eps_adtrain
        eps_step = args.eps_step_adtrain
        max_iter_adtrain = args.max_iter_adtrain
    model_name = dataset + '_' + fil + fil_mode + str(fil_thresh) + '_' + train_method + str(clean_adv) + reg_loss + reg_norm + reg_mode_dct + str(alpha) + '_epoch' + str(n_epochs)  + "_" + 'batch' + str(batch_size) + \
                         '_eps_' + str(eps) + '_step_' + str(eps_step) + '_maxite_' + str(max_iter_adtrain) + 'coe_reg_time_' + str(reg_coe_time) + '_' + str(reg_coe_dct)
    fname = 'model_adtrain/' + 'bestmodel_' + model_name + '.h5'
    fname_adv = 'model_adtrain/' + 'bestmodel_adv_' + model_name + '.h5'
    log_name = 'adtrain_log/' + model_name + '.log'

    classifier_adv = KerasClassifier(model=model_adv, use_logits=False, clip_values=(0.0, 1.0))
    attack = ProjectedGradientDescent(estimator=classifier_adv, eps=eps, eps_step=eps_step, max_iter=max_iter_adtrain, batch_size=batch_size, verbose=False)
    if dataset == 'mnist':
        eps2 = args.eps_attack_mnist
        eps_step2 = args.eps_step_attack_mnist
        max_iter = args.max_iter_attack_mnist
    else:
        eps2 = args.eps_attack
        eps_step2 = args.eps_step_attack
        max_iter = args.max_iter_attack
    attack2 = ProjectedGradientDescent(estimator=classifier_adv, eps=eps2, eps_step=eps_step2, max_iter=max_iter, batch_size=batch_size, verbose=False)

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG,
        filename=log_name
    )
    logger.info(args)
    logger.info(
        'Epoch \t TraTime \t TraLoss \t TraAcc \t TraLossAdv \t TraAccAdv \t TesLoss \t TesAcc \t TesLossAdv \t Losstime \t TesAccAdv')

    index = index_train
    test_adv_best = copy.deepcopy(x_test)
    for ep in range(n_epochs):
        start_time = time.time()
        print(colored("this is epoch: {0}\n", 'red').format(ep))
        if ep < 100:
            K.set_value(model.optimizer.learning_rate, lr)
        elif ep > 99 and ep< 150:
            K.set_value(model.optimizer.learning_rate, lr/10)
        else:
            K.set_value(model.optimizer.learning_rate, lr/100)
        # print("Learning rate:", K.eval(model.optimizer.learning_rate))
        batchs = []
        n_batch = math.ceil(num_train / batch_size)
        np.random.shuffle(copy.deepcopy(index))

        for bat in range(n_batch):
            if bat < n_batch - 1:
                batch = index[bat * batch_size: (bat + 1) * batch_size]
                batch = np.array(batch)
                batchs.append(batch)
            else:
                batch = index[bat * batch_size:]
                batch = np.array(batch)
                batchs.append(batch)

        for epb in range(n_batch - 1):
            batch_y = copy.deepcopy(y_train[batchs[epb]])
            batch_clean = copy.deepcopy(x_train[batchs[epb]])
            if clean_adv[0] == 1:
                model.fit([batch_clean, batch_y], batch_y, epochs=1, shuffle=False)
            if clean_adv[1] == 1:
                batch_adv = attack.generate(x=batch_clean, y=batch_y)
                dif = np.sum(np.abs(batch_adv - batch_clean))
                print('this is diff between clean and adv:', dif)
                model.fit([batch_adv, batch_y], batch_y, epochs=1, shuffle=False)

        x_sample = copy.deepcopy(x_train[0:1000])
        y_sample = copy.deepcopy(y_train[0:1000])
        x_adv = attack2.generate(x=x_sample, y=y_sample)
        x_pred_clean = model_adv.predict(x_sample)
        x_pred_adv = model_adv.predict(x_adv)
        x_loss_clean = cat_cross(x_pred_clean, y_sample)
        x_loss_adv = cat_cross(x_pred_adv, y_sample)
        yPred_clean = to_categorical(np.argmax(x_pred_clean, axis=1), num_classes=10)
        x_acc_clean = metrics.accuracy_score(y_sample, yPred_clean)
        yPred_adv = to_categorical(np.argmax(x_pred_adv, axis=1), num_classes=10)
        x_acc_adv = metrics.accuracy_score(y_sample, yPred_adv)
        train_time = time.time()


        val_adv = attack2.generate(x=copy.deepcopy(x_val), y=copy.deepcopy(y_val))
        val_pred_clean = model_adv.predict(x_val)
        val_pred_adv = model_adv.predict(val_adv)
        val_loss_clean = cat_cross(val_pred_clean, y_val)
        val_loss_adv = cat_cross(val_pred_adv, y_val)
        yPred_clean = to_categorical(np.argmax(val_pred_clean, axis=1), num_classes=10)
        val_acc_clean = metrics.accuracy_score(y_val, yPred_clean)
        yPred_adv = to_categorical(np.argmax(val_pred_adv, axis=1), num_classes=10)
        val_acc_adv = metrics.accuracy_score(y_val, yPred_adv)

        test_adv = attack2.generate(x=copy.deepcopy(x_test), y=copy.deepcopy(y_test))
        pred_pro_clean = model_adv.predict(x_test)
        pred_pro_adv = model_adv.predict(test_adv)
        test_loss_clean = cat_cross(pred_pro_clean, y_test)
        test_loss_adv = cat_cross(pred_pro_adv, y_test)
        yPred_clean = to_categorical(np.argmax(pred_pro_clean, axis=1), num_classes=10)
        accuracy_clean = metrics.accuracy_score(y_test, yPred_clean)
        yPred_adv = to_categorical(np.argmax(pred_pro_adv, axis=1), num_classes=10)
        accuracy_adv = metrics.accuracy_score(y_test, yPred_adv)

        # loss_grad_time = fun_grad_time.predict([x_test, y_test])
        # loss_grad_dct = fun_grad_dct.predict([x_test, y_test])
        loss_grad_time = np.mean(np.abs(fun_grad_time.predict([x_sample, y_sample])))

        logger.info(
            '%d \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f  \t %.4f \t %.4f',
            ep, train_time - start_time,
            x_loss_clean, x_acc_clean, x_loss_adv, x_acc_adv,
            test_loss_clean, accuracy_clean, test_loss_adv, loss_grad_time, accuracy_adv)

        logger.info('validation %.4f \t %.4f \t %.4f \t %.4f',
                    val_loss_clean, val_acc_clean, val_loss_adv, val_acc_adv)

        if val_acc_adv > val_acc_best:
            val_acc_best = val_acc_adv
            model.save_weights(fname)
            model_adv.save_weights(fname_adv)
            test_adv_best = test_adv
    adv_name = 'adv_best/adv' + model_name + '.npy'
    np.save(adv_name, test_adv_best)

if __name__ == "__main__":
    main()