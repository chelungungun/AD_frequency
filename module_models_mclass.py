"""
This file is to build the models needs by multi-class classification tasks
"""
from tensorflow.keras.layers import Input , Dense , Flatten , Activation , Conv2D , Lambda, MaxPooling2D, UpSampling2D, AveragePooling2D, Concatenate
from tensorflow.keras import Model
import wide_residual_network as wrn
import wide_residual_network_back2 as wrn_b2
from module_autoencoder_mc import auto_conv_time_hidden, auto_conv_time_noh, auto_conv_dct_noh, auto_conv_dct_hidden
import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np
import math
from tensorflow.keras.losses import MSE, categorical_crossentropy


# from keras.layers import Input , Dense , Flatten , Activation , Conv2D , Lambda, MaxPooling2D, UpSampling2D, AveragePooling2D, Concatenate
# from keras import Model
# import wide_residual_network as wrn
# import wide_residual_network_back2 as wrn_b2
# from module_autoencoder_mc import auto_conv_time_hidden, auto_conv_time_noh, auto_conv_dct_noh, auto_conv_dct_hidden
# import keras.backend as K
# import tensorflow as tf
# import numpy as np
# import math
# from keras.losses import MSE, categorical_crossentropy

# def loss_reg_gradient_time(x):
#     # x[0]:outputs, x[1]: inputs, x[2]:y
#     condition = tf.equal(x[2], 1)
#     indices = tf.where(condition)
#     one_out = tf.gather_nd(x[0], indices=indices)
#     grads = tf.gradients(one_out , x[1])[0]
#     loss = K.mean(tf.abs(grads))
#
#     return loss

def reg_gradient_time_mse_l1(x):
    # x[0]:outputs, x[1]: inputs, x[2]:y
    cat_loss = MSE(x[2], x[0])
    grads = tf.gradients(cat_loss, x[1], unconnected_gradients='zero')[0]
    loss = K.mean(K.abs(grads))

    return loss

def reg_gradient_time_mse_l2(x):
    # x[0]:outputs, x[1]: inputs, x[2]:y
    cat_loss = MSE(x[2], x[0])
    grads = tf.gradients(cat_loss, x[1], unconnected_gradients='zero')[0]
    loss = K.mean(K.square(grads))

    return loss

def reg_gradient_time_entropy_l1(x):
    # x[0]:outputs, x[1]: inputs, x[2]:y
    cat_loss = categorical_crossentropy(x[2], x[0])
    grads = tf.gradients(cat_loss, x[1], unconnected_gradients='zero')[0]
    loss = K.mean(K.abs(grads))

    return loss

def reg_gradient_time_entropy_l2(x):
    # x[0]:outputs, x[1]: inputs, x[2]:y
    cat_loss = categorical_crossentropy(x[2], x[0])
    grads = tf.gradients(cat_loss, x[1], unconnected_gradients='zero')[0]
    loss = K.mean(K.square(grads))

    return loss

def reg_gradient_dct_mse_l1(x, weights):
    # x[0]:outputs, x[1]: inputs, x[2]:y
    cat_loss = MSE(x[2], x[0])
    grads = tf.gradients(cat_loss, x[1], unconnected_gradients='zero')[0]
    X1 = tf.transpose(grads, perm=[0, 3, 1, 2])
    X2 = tf.signal.dct(X1, type=2, norm='ortho')
    X2_t = tf.transpose(X2, perm=[0, 1, 3, 2])
    X3 = tf.signal.dct(X2_t, type=2, norm='ortho')
    X3_t = tf.transpose(X3, perm=[0, 3, 2, 1])
    loss = K.mean(K.abs(X3_t) * weights)

    return loss

def reg_gradient_dct_mse_l2(x, weights):
    # x[0]:outputs, x[1]: inputs, x[2]:y
    cat_loss = MSE(x[2], x[0])
    grads = tf.gradients(cat_loss, x[1], unconnected_gradients='zero')[0]
    X1 = tf.transpose(grads, perm=[0, 3, 1, 2])
    X2 = tf.signal.dct(X1, type=2, norm='ortho')
    X2_t = tf.transpose(X2, perm=[0, 1, 3, 2])
    X3 = tf.signal.dct(X2_t, type=2, norm='ortho')
    X3_t = tf.transpose(X3, perm=[0, 3, 2, 1])
    loss = K.mean(K.square(X3_t) * weights)

    return loss

def reg_gradient_dct_entropy_l1(x, weights):
    # x[0]:outputs, x[1]: inputs, x[2]:y
    cat_loss = categorical_crossentropy(x[2], x[0])
    grads = tf.gradients(cat_loss, x[1], unconnected_gradients='zero')[0]
    X1 = tf.transpose(grads, perm=[0, 3, 1, 2])
    X2 = tf.signal.dct(X1, type=2, norm='ortho')
    X2_t = tf.transpose(X2, perm=[0, 1, 3, 2])
    X3 = tf.signal.dct(X2_t, type=2, norm='ortho')
    X3_t = tf.transpose(X3, perm=[0, 3, 2, 1])
    loss = K.mean(K.abs(X3_t) * weights)

    return loss

def reg_gradient_dct_entropy_l2(x, weights):
    # x[0]:outputs, x[1]: inputs, x[2]:y
    cat_loss = categorical_crossentropy(x[2], x[0])
    grads = tf.gradients(cat_loss, x[1], unconnected_gradients='zero')[0]
    X1 = tf.transpose(grads, perm=[0, 3, 1, 2])
    X2 = tf.signal.dct(X1, type=2, norm='ortho')
    X2_t = tf.transpose(X2, perm=[0, 1, 3, 2])
    X3 = tf.signal.dct(X2_t, type=2, norm='ortho')
    X3_t = tf.transpose(X3, perm=[0, 3, 2, 1])
    loss = K.mean(K.square(X3_t) * weights)

    return loss

def reg_gradient_dct_mse_l1_b2(x, weights):
    # x[0]:outputs, x[1]: out_back2, x[2]:y
    cat_loss = MSE(x[2], x[0])
    grads = tf.gradients(cat_loss, x[1], unconnected_gradients='zero')[0]
    X2 = tf.signal.dct(grads, type=2, norm='ortho')
    loss = K.mean(K.abs(X2) * weights)

    return loss

def reg_gradient_dct_mse_l2_b2(x, weights):
    # x[0]:outputs, x[1]: inputs, x[2]:y
    cat_loss = MSE(x[2], x[0])
    grads = tf.gradients(cat_loss, x[1], unconnected_gradients='zero')[0]
    X2 = tf.signal.dct(grads, type=2, norm='ortho')
    loss = K.mean(K.square(X2) * weights)

    return loss

def reg_gradient_dct_entropy_l1_b2(x, weights):
    # x[0]:outputs, x[1]: inputs, x[2]:y
    cat_loss = categorical_crossentropy(x[2], x[0])
    grads = tf.gradients(cat_loss, x[1], unconnected_gradients='zero')[0]
    X2 = tf.signal.dct(grads, type=2, norm='ortho')
    loss = K.mean(K.abs(X2) * weights)

    return loss

def reg_gradient_dct_entropy_l2_b2(x, weights):
    # x[0]:outputs, x[1]: inputs, x[2]:y
    cat_loss = categorical_crossentropy(x[2], x[0])
    grads = tf.gradients(cat_loss, x[1], unconnected_gradients='zero')[0]
    X2 = tf.signal.dct(grads, type=2, norm='ortho')
    loss = K.mean(K.square(X2) * weights)

    return loss

def grads_mse(x):
    # x[0]:outputs, x[1]: inputs, x[2]:y
    cat_loss = MSE(x[2], x[0])
    grads = tf.gradients(cat_loss, x[1], unconnected_gradients='zero')[0]

    return grads

def grads_entropy(x):
    # x[0]:outputs, x[1]: inputs, x[2]:y
    cat_loss = categorical_crossentropy(x[2], x[0])
    grads = tf.gradients(cat_loss, x[1], unconnected_gradients='zero')[0]

    return grads


def spectrum_one(x, k, p):
    ## this fun is to calculate the mag of spectrum k
    ## x[0]: outputs x[1]:inputs; k: freq; p: the projection vector, the main component of PCA
    volumeSize = K.int_shape(x[1])

    x_flat = tf.reshape(x[1], shape=[tf.shape(x[1])[0], np.prod(volumeSize[1:])])
    # x_flat = tf.reshape(x[1], shape=[128, 784])

    x_exp = tf.exp(tf.dtypes.complex(real=0.0, imag=tf.linalg.matvec(x_flat, tf.constant(p)) * (-2 * k * math.pi)))
    real = tf.math.real(x_exp)
    img = tf.math.imag(x_exp)

    y_k_real= tf.linalg.matvec(tf.transpose(x[0]), real)# n_class *
    y_k_img= tf.linalg.matvec(tf.transpose(x[0]), img) # n_class *
    y_k_mag = tf.abs(tf.dtypes.complex(real=y_k_real, imag=y_k_img))
    # y_k_mag = tf.sqrt(y_k_real^2 + y_k_img^2)

    y_K_mean = K.expand_dims(K.mean(y_k_mag), axis=-1)

    return y_K_mean


def loss_reg_output(x, weights):
    # x: dft_mag of each spectrum (batch_size * ), weights: batch_size *
    loss = K.sum(x * weights)
    return loss


def small_cnn_reg_out(inputshape_x, loss_coe, weights, batch_size, p):
    input = Input(shape=inputshape_x)
    x = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform')(input)
    x = MaxPooling2D()(x)
    x = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform')(x)
    # x = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform')(x)
    x = MaxPooling2D()(x)
    x = Flatten()(x)
    x = Dense(100, activation='relu', kernel_initializer='he_uniform')(x)
    output = Dense(10, activation='softmax')(x)
    layer_spectrum_one = Lambda(spectrum_one)
    y_k_mags = []
    for k in range(batch_size):
        layer_spectrum_one.arguments = {'k': k, 'p':p}
        y_k_mag = layer_spectrum_one([output, input])
        y_k_mags.append(y_k_mag)
    dft_mags = Concatenate()(y_k_mags)

    layer_loss_out = Lambda(loss_reg_output)
    layer_loss_out.arguments = {'weights': weights}
    loss_out = layer_loss_out(dft_mags) * loss_coe
    model = Model(input, output)
    model.add_loss(loss_out)

    # #build model with single input for attack
    # model_adv = Model(input, output)

    return model, model


def small_cnn_reg_grad(inputshape_x, inputshape_y, reg_coe_time, reg_coe_dct, reg_norm, reg_loss, weights):
    input = Input(shape=inputshape_x)
    y = Input(shape=inputshape_y)
    x = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform')(input)
    x = MaxPooling2D()(x)
    x = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform')(x)
    # x = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform')(x)
    x = MaxPooling2D()(x)
    x = Flatten()(x)
    x = Dense(100, activation='relu', kernel_initializer='he_uniform')(x)
    h = Dense(10, activation='linear')(x)
    output = Activation('softmax')(h)
    if reg_norm == 'l1' and reg_loss == 'mse':
        loss_layer_time = Lambda(reg_gradient_time_mse_l1)
        loss_layer_dct = Lambda(reg_gradient_dct_mse_l1_b2)
    elif reg_norm == 'l2' and reg_loss == 'mse':
        loss_layer_time = Lambda(reg_gradient_time_mse_l2)
        loss_layer_dct = Lambda(reg_gradient_dct_mse_l2_b2)
    elif reg_norm == 'l1' and reg_loss == 'entropy':
        loss_layer_time = Lambda(reg_gradient_time_entropy_l1)
        loss_layer_dct = Lambda(reg_gradient_dct_entropy_l1_b2)
    else:
        loss_layer_time = Lambda(reg_gradient_time_entropy_l2)
        loss_layer_dct = Lambda(reg_gradient_dct_entropy_l2_b2)

    loss_time = loss_layer_time([output, x, y]) * reg_coe_time
    loss_layer_dct.arguments = {'weights': weights}
    loss_dct = loss_layer_dct([output, x, y]) * reg_coe_dct

    model = Model([input, y], output)
    model.add_loss(loss_time + loss_dct)

    #build model with single input for attack
    if reg_loss == 'mse':
        grad_layer = Lambda(grads_mse)
    else:
        grad_layer = Lambda(grads_entropy)
    grads_time = grad_layer([output, input, y])
    fun_grad_time = Model([input, y], grads_time)
    model_adv = Model(input, output)

    return model_adv, model, fun_grad_time

def small_cnn(inputshape):
    input = Input(shape=inputshape)
    x = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform')(input)
    x = MaxPooling2D()(x)
    x = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform')(x)
    # x = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform')(x)
    x = MaxPooling2D()(x)
    x = Flatten()(x)
    x = Dense(100, activation='relu', kernel_initializer='he_uniform')(x)
    output = Dense(10, activation='softmax')(x)
    model = Model(input, output)

    return model


def grads_h(x):
    # x[0]:outputs, x[1]: inputs, x[2]:y
    condition = tf.equal(x[2], 1)
    indices = tf.where(condition)
    one_out = tf.gather_nd(x[0], indices=indices)
    grads = tf.gradients(one_out, x[1])[0]

    return grads


def grads_loss(x):
    # x[0]:outputs, x[1]: inputs, x[2]:y
    cat_loss = tf.keras.losses.CategoricalCrossentropy()(x[2], x[0])
    # cat_loss = tf.keras.losses.MSE(x[2], x[0])
    grads = tf.gradients(cat_loss, x[1])[0]

    return grads

def small_cnn_grad(inputshape_x, inputshape_y):
    input = Input(shape=inputshape_x)
    y = Input(shape=inputshape_y)
    x = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform')(input)
    x = MaxPooling2D()(x)
    x = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform')(x)
    # x = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform')(x)
    x = MaxPooling2D()(x)
    x = Flatten()(x)
    x = Dense(100, activation='relu', kernel_initializer='he_uniform')(x)
    h = Dense(10, activation='linear')(x)
    output = Activation('softmax')(h)

    layer_grad_h = Lambda(grads_h)
    grad_h = layer_grad_h([h, input, y])

    layer_grad_loss = Lambda(grads_loss)
    grad_loss = layer_grad_loss([output, input, y])

    fun_grad_h = Model([input, y], grad_h)
    fun_grad_loss = Model([input, y], grad_loss)
    model = Model(input, output)

    return model, fun_grad_h, fun_grad_loss

# def wrn_cifar10_reg_grad(inputshape_x, inputshape_y, loss_coe, weights):
#     ## return wrn with nornalized input
#     inputs = Input(shape=inputshape_x)
#     y = Input(shape=inputshape_y)
#
#     # layer_normlize = Lambda(normlize)
#     # inputs = layer_normlize(inputs)
#     WRN = wrn.create_wide_residual_network((32, 32, 3), nb_classes=10, N=2, k=8, dropout=0.0)
#     outputs = WRN(inputs)
#     loss_layer = Lambda(loss_reg_gradient)
#     loss_layer.arguments = {'weights': weights}
#     loss = loss_layer([outputs, inputs, y]) * loss_coe
#     model = Model([inputs,y], outputs)
#     model.add_loss(loss)
#
#     #build model with single input for attack
#     model_adv = Model(inputs, outputs)
#
#     return model_adv, model


def wrn_cifar10_reg_grad(inputshape_x, inputshape_y, reg_coe_time, reg_coe_dct, reg_norm, reg_loss, weights):
    ## return wrn with nornalized input
    input = Input(shape=inputshape_x)
    y = Input(shape=inputshape_y)
    layer_normlize = Lambda(normlize)
    input_norm = layer_normlize(input)
    WRN = wrn.create_wide_residual_network((32, 32, 3), nb_classes=10, N=2, k=8, dropout=0.0)
    output = WRN(input_norm)

    if reg_norm == 'l1' and reg_loss == 'mse':
        loss_layer_time = Lambda(reg_gradient_time_mse_l1)
        loss_layer_dct = Lambda(reg_gradient_dct_mse_l1)
    elif reg_norm == 'l2' and reg_loss == 'mse':
        loss_layer_time = Lambda(reg_gradient_time_mse_l2)
        loss_layer_dct = Lambda(reg_gradient_dct_mse_l2)
    elif reg_norm == 'l1' and reg_loss == 'entropy':
        loss_layer_time = Lambda(reg_gradient_time_entropy_l1)
        loss_layer_dct = Lambda(reg_gradient_dct_entropy_l1)
    else:
        loss_layer_time = Lambda(reg_gradient_time_entropy_l2)
        loss_layer_dct = Lambda(reg_gradient_dct_entropy_l2)

    loss_time = loss_layer_time([output, input, y]) * reg_coe_time
    loss_layer_dct.arguments = {'weights': weights}
    loss_dct = loss_layer_dct([output, input, y]) * reg_coe_dct
    model = Model([input, y], output)
    model.add_loss(loss_time + loss_dct)

    #build model with single input for attack
    if reg_loss == 'mse':
        grad_layer = Lambda(grads_mse)
    else:
        grad_layer = Lambda(grads_entropy)
    grads_time = grad_layer([output, input, y])
    fun_grad_time = Model([input, y], grads_time)
    model_adv = Model(input, output)

    return model_adv, model, fun_grad_time


def wrn_cifar10_reg_grad_back2(inputshape_x, inputshape_y, reg_coe_time, reg_coe_dct, reg_norm, reg_loss, weights):
    ## return wrn with nornalized input
    input = Input(shape=inputshape_x)
    y = Input(shape=inputshape_y)
    layer_normlize = Lambda(normlize)
    input_norm = layer_normlize(input)
    # WRN = wrn_b2.create_wide_residual_network((32, 32, 3), nb_classes=10, N=5, k=10, dropout=0.0)
    WRN = wrn_b2.create_wide_residual_network((32, 32, 3), nb_classes=10, N=2, k=8, dropout=0.0)
    output_2 = WRN(input_norm)
    output = Dense(10, activation='softmax')(output_2)

    if reg_norm == 'l1' and reg_loss == 'mse':
        loss_layer_time = Lambda(reg_gradient_time_mse_l1)
        loss_layer_dct = Lambda(reg_gradient_dct_mse_l1_b2)
    elif reg_norm == 'l2' and reg_loss == 'mse':
        loss_layer_time = Lambda(reg_gradient_time_mse_l2)
        loss_layer_dct = Lambda(reg_gradient_dct_mse_l2_b2)
    elif reg_norm == 'l1' and reg_loss == 'entropy':
        loss_layer_time = Lambda(reg_gradient_time_entropy_l1)
        loss_layer_dct = Lambda(reg_gradient_dct_entropy_l1_b2)
    else:
        loss_layer_time = Lambda(reg_gradient_time_entropy_l2)
        loss_layer_dct = Lambda(reg_gradient_dct_entropy_l2_b2)

    loss_time = loss_layer_time([output, output_2, y]) * reg_coe_time
    loss_layer_dct.arguments = {'weights': weights}
    loss_dct = loss_layer_dct([output, output_2, y]) * reg_coe_dct
    model = Model([input, y], output)
    model.add_loss(loss_time + loss_dct)

    #build model with single input for attack
    if reg_loss == 'mse':
        grad_layer = Lambda(grads_mse)
    else:
        grad_layer = Lambda(grads_entropy)
    grads_time = grad_layer([output, input, y])
    fun_grad_time = Model([input, y], grads_time)
    model_adv = Model(input, output)

    return model_adv, model, fun_grad_time


def wrn_cifar10_reg_out(inputshape_x, loss_coe, weights, batch_size, p):
    ## return wrn with nornalized input
    inputs = Input(shape=inputshape_x)
    # layer_normlize = Lambda(normlize)
    # inputs = layer_normlize(inputs)

    WRN = wrn.create_wide_residual_network((32, 32, 3), nb_classes=10, N=2, k=8, dropout=0.0)
    outputs = WRN(inputs)
    layer_spectrum_one = Lambda(spectrum_one)
    y_k_mags = []
    for k in range(batch_size):
        layer_spectrum_one.arguments = {'k': k, 'p': p}
        y_k_mag = layer_spectrum_one([outputs, input])
        y_k_mags.append(y_k_mag)
    dft_mags = Concatenate()(y_k_mags)

    layer_loss_out = Lambda(loss_reg_output)
    layer_loss_out.arguments = {'weights': weights}
    loss_out = layer_loss_out(dft_mags) * loss_coe
    model = Model(inputs, outputs)
    model.add_loss(loss_out)

    # #build model with single input for attack
    # model_adv = Model(inputs, outputs)

    return model


def wrn_cifar10():
    ## return WRN-34-10 for cifar10
    WRN = wrn.create_wide_residual_network((32, 32, 3), nb_classes=10, N=2, k=8, dropout=0.0)
    # model = wrn.create_wide_residual_network((32, 32, 3), nb_classes=10, N=2, k=4, dropout=0.0)

    return WRN

def wrn_cifar10_norm(inputshape):
    ## return wrn with nornalized input
    inputs = Input(shape=inputshape)
    layer_normlize = Lambda(normlize)
    inputs_norm = layer_normlize(inputs)
    WRN = wrn.create_wide_residual_network((32, 32, 3), nb_classes=10, N=2, k=8, dropout=0.0)
    outputs = WRN(inputs_norm)
    WRN_norm = Model(inputs, outputs)

    return WRN_norm


def reshape(x, inputshape):
    re_x = x[:,0:inputshape[0],0:inputshape[1],:]

    return re_x

def loss_rebuild_time(sp_input_rebuild):
    ## build the reconstruction error of autoencoder
    ## sp_input_rebuild: lists[4]: sp_input_real, imag, sp_rebuild_real, imag
    loss = K.mean(K.square(sp_input_rebuild[0] - sp_input_rebuild[1]))

    return loss


def loss_rebuild_adv_time(sp_clean_adv):
    ## build the error between reconstructed clean spectrum and perturbed spectrum
    ## sp_clean_adv: lists[4]: sp_clean_real, imag, sp_adv_real, imag
    loss = K.mean(K.square(sp_clean_adv[0] - sp_clean_adv[1]))

    return loss


def normlize(x):
    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2471, 0.2435, 0.2616)
    x2 = (x - cifar10_mean) / cifar10_std

    return x2

def model_time_rebuild(model_cla, inputshape, mode, h_diem, loss_coe_rebuild, loss_coe_adv, filters):
    inputs_clean = Input(shape=inputshape) #
    inputs_adv = Input(shape=inputshape)

    #select different autoencoders by parameter mode
    if mode == 'noh1' or mode == 'noh2':
        layer_AT_mlp_real = auto_conv_time_noh(inputshape=inputshape, filters=filters, mode=mode)
    else:
        layer_AT_mlp_real = auto_conv_time_hidden(inputshape=inputshape, filters=filters, h_diem=h_diem)
    # recon_clean = layer_AT_mlp_real(inputs_clean)
    recon_adv = layer_AT_mlp_real(inputs_adv)

    layer_shape = Lambda(reshape)
    layer_shape.arguments = {'inputshape': inputshape}
    # recon_clean = layer_shape(recon_clean)
    recon_adv = layer_shape(recon_adv)

    layer_normlize = Lambda(normlize)

    if model_cla == 'mnist':
        base_model = small_cnn(inputshape=inputshape)
        recon_adv_norm = recon_adv
    else:
        base_model = wrn_cifar10()
        recon_adv_norm = layer_normlize(recon_adv)
        # recon_clean = Permute(dims=(0, 2,3,1))(recon_clean)
        # recon_adv = Permute(dims=(0, 2,3,1))(recon_adv)

    # outputs_clean = base_model(recon_clean)
    outputs_adv = base_model(recon_adv_norm)

    loss_clean_sprecon = Lambda(loss_rebuild_time)
    loss_adv_sprecon = Lambda(loss_rebuild_adv_time)
    # loss_clean_value = loss_clean_sprecon([inputs_clean, recon_clean]) * loss_coe_rebuild
    loss_adv_value = loss_adv_sprecon([inputs_clean, recon_adv]) * loss_coe_adv
    # model = Model(inputs = [inputs_clean, inputs_adv], outputs = [outputs_clean, outputs_adv])
    model = Model(inputs = [inputs_clean, inputs_adv], outputs = outputs_adv)

    # model.add_loss([loss_clean_value, loss_adv_value])
    model.add_loss(loss_adv_value)

    #This model is for the generation of samples for adversarial training
    model_adv = Model(inputs_adv, outputs_adv)

    # This function is to output the reconstructed fft signals
    model_recon_time = Model(inputs_adv, recon_adv)

    return model, model_adv, model_recon_time


def dct_input(X):
    X1 = tf.transpose(X, perm=[0, 3, 1, 2])
    X2 = tf.signal.dct(X1, type=2, norm='ortho')
    X2_t = tf.transpose(X2, perm=[0, 1, 3, 2])
    X3 = tf.signal.dct(X2_t, type=2, norm='ortho')
    X3_t = tf.transpose(X3, perm=[0, 3, 2, 1])
    return X3_t


def idct_rebuild(X):
    X1 = tf.transpose(X, perm=[0, 3, 1, 2])
    X2 = tf.signal.idct(X1, type=2, norm='ortho')
    X2_t = tf.transpose(X2, perm=[0, 1, 3, 2])
    X3 = tf.signal.idct(X2_t, type=2, norm='ortho')
    X3_t = tf.transpose(X3, perm=[0, 3, 2, 1])
    return X3_t


def model_dct_rebuild(model_cla, inputshape, mode, h_diem, loss_coe_rebuild, loss_coe_adv, filters):
    inputs_clean = Input(shape=inputshape) #
    inputs_adv = Input(shape=inputshape)

    layer_dct = Lambda(dct_input)
    inputs_clean_dct = layer_dct(inputs_clean)  # batch * 3 * ? * ?
    inputs_adv_dct = layer_dct(inputs_adv)

    #select different autoencoders by parameter mode
    if mode == 'noh1' or mode == 'noh2':
        layer_AT_mlp_real = auto_conv_dct_noh(inputshape=inputshape, filters=filters, mode=mode)
    else:
        layer_AT_mlp_real = auto_conv_dct_hidden(inputshape=inputshape, filters=filters, h_diem=h_diem, mode=mode)
    # recon_cleandct = layer_AT_mlp_real(inputs_clean)
    recon_adv_dct = layer_AT_mlp_real(inputs_adv_dct)

    layer_shape = Lambda(reshape)
    layer_shape.arguments = {'inputshape': inputshape}
    # recon_clean_dct = layer_shape(recon_clean_dct)
    recon_adv_dct = layer_shape(recon_adv_dct)

    layer_idct = Lambda(idct_rebuild)
    # recon_clean = layer_idct(recon_clean_dct)
    recon_adv = layer_idct(recon_adv_dct)

    if model_cla == 'mnist':
        base_model = small_cnn(inputshape=inputshape)
    else:
        base_model = wrn_cifar10()
        # recon_clean = Permute(dims=(0, 2,3,1))(recon_clean)
        # recon_adv = Permute(dims=(0, 2,3,1))(recon_adv)

    # outputs_clean = base_model(recon_clean)
    outputs_adv = base_model(recon_adv)

    # loss_clean_sprecon = Lambda(loss_rebuild_time)
    loss_adv_sprecon = Lambda(loss_rebuild_adv_time)
    # loss_clean_value = loss_clean_sprecon([inputs_clean, recon_clean]) * loss_coe_rebuild
    loss_adv_value = loss_adv_sprecon([inputs_clean_dct, recon_adv_dct]) * loss_coe_adv
    # model = Model(inputs = [inputs_clean, inputs_adv], outputs = [outputs_clean, outputs_adv])
    model = Model(inputs = [inputs_clean, inputs_adv], outputs = outputs_adv)

    # model.add_loss([loss_clean_value, loss_adv_value])
    model.add_loss(loss_adv_value)

    #This model is for the generation of samples for adversarial training
    model_adv = Model(inputs_adv, outputs_adv)

    # This function is to output the reconstructed fft signals
    model_recon_dct = Model(inputs_adv, recon_adv_dct)

    return model, model_adv, model_recon_dct

