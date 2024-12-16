import numpy as np
from matplotlib import pyplot as plt

import tensorflow as tf

import os
import time
import datetime
import random
import math
import pandas as pd

from IPython import display

from sklearn.model_selection import train_test_split

from scipy.signal import find_peaks
from scipy.signal import savgol_filter

from kneed import KneeLocator

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.compat.v1.enable_eager_execution()

data_window_length = 400
data_overlap = 0.9
input_length = 200
prediction_length = 10
input_windows = int(input_length / prediction_length)
disc_future = 20
num_prediction_vars = 4
num_ancilliary_vars = 4

train_steps = 30000
mini_batch_size = 1
encoder_dropout = 0.2
l1_loss_weight = 10
gen_learning_rate = 0.0001
disc_learning_rate = 0.0001

result_dir = '/example_results_path/'
data_set_dir = '/example_data_set_path'

if not os.path.exists(result_dir):
    os.makedirs(result_dir)
with open(result_dir + "/config.txt", "w") as config_file:
    config_file.write("data_window_length = " + str(data_window_length) +
                      "\ninput_length = " + str(input_length) +
                      "\nprediction_length = " + str(prediction_length) +
                      "\ndisc_future = " + str(disc_future) +
                      "\nnum_prediction_vars = " + str(num_prediction_vars) +
                      "\nnum_ancilliary_vars = " + str(num_ancilliary_vars) +
                      "\nnum_pairs = " + str(num_pairs) +
                      "\nfilter_by_phase = " + str(filter_by_phase) +
                      "\nfilter_by_intervention = " + str(filter_by_intervention) +
                      "\nfilter_by_animal = " + str(filter_by_animal) +
                      "\nintervention_filter = " + str(intervention_filter) +
                      "\nanimal_filter = " + str(animal_filter) +
                      "\ntrain_steps = " + str(train_steps) +
                      "\nmini_batch_size = " + str(mini_batch_size) +
                      "\nencoder_dropout = " + str(encoder_dropout) +
                      "\nl1_loss_weight = " + str(l1_loss_weight) +
                      "\ngen_learning_rate = " + str(gen_learning_rate) +
                      "\ndisc_learning_rate = " + str(disc_learning_rate)
                      )

# load data set
dataset = pd.read_pickle(data_set_dir)
x = dataset


# cut data to fit model
def split_tf(data):
    # split off ancilliary vars like intervention, phase, and animal
    splitteded = tf.split(data, [num_prediction_vars, num_ancilliary_vars], axis=0)
    splitted_disc = tf.split(splitteded[0], [input_length + (disc_future * prediction_length),
                                             data_window_length - (input_length + (disc_future * prediction_length))],
                             axis=1)

    # split into input and target image and create overlap
    splitted_first = tf.split(splitteded[0],
                              [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
                               10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
                               10, 10, 10, 10, 10, 10, 10, 10],
                              axis=1)

    vad_speed = splitteded[1][-4]
    latent_phase = splitteded[1][-3][0]
    latent_inter = splitteded[1][-1][0]

    return splitted_first[0], splitted_first[1], splitted_first[2], splitted_first[3], splitted_first[4], \
           splitted_first[5], splitted_first[6], splitted_first[7], splitted_first[8], splitted_first[9], \
           splitted_first[10], splitted_first[11], splitted_first[12], splitted_first[13], splitted_first[14], \
           splitted_first[15], splitted_first[16], splitted_first[17], splitted_first[18], splitted_first[19], \
           splitted_first[20], splitted_first[21], splitted_first[22], splitted_first[23], splitted_first[24], \
           splitted_first[25], splitted_first[26], splitted_first[27], splitted_first[28], splitted_first[29], \
           splitted_first[30], splitted_first[31], splitted_first[32], splitted_first[33], splitted_first[34], \
           splitted_first[35], splitted_first[36], splitted_first[37], splitted_first[38], splitted_first[39], \
           tf.expand_dims(vad_speed[0:200], axis=0), \
           tf.expand_dims(vad_speed[10:210], axis=0), \
           tf.expand_dims(vad_speed[20:220], axis=0), \
           tf.expand_dims(vad_speed[30:230], axis=0), \
           tf.expand_dims(vad_speed[40:240], axis=0), \
           tf.expand_dims(vad_speed[50:250], axis=0), \
           tf.expand_dims(vad_speed[60:260], axis=0), \
           tf.expand_dims(vad_speed[70:270], axis=0), \
           tf.expand_dims(vad_speed[80:280], axis=0), \
           tf.expand_dims(vad_speed[90:290], axis=0), \
           tf.expand_dims(vad_speed[100:300], axis=0), \
           tf.expand_dims(vad_speed[110:310], axis=0), \
           tf.expand_dims(vad_speed[120:320], axis=0), \
           tf.expand_dims(vad_speed[130:330], axis=0), \
           tf.expand_dims(vad_speed[140:340], axis=0), \
           tf.expand_dims(vad_speed[150:350], axis=0), \
           tf.expand_dims(vad_speed[160:360], axis=0), \
           tf.expand_dims(vad_speed[170:370], axis=0), \
           tf.expand_dims(vad_speed[180:380], axis=0), \
           tf.expand_dims(vad_speed[190:390], axis=0), splitted_disc[0], splitteded[0], latent_phase, latent_inter


# split into test, validation, and training sets
x_train, x_test, _, _ = train_test_split(x, x, test_size=0.1)
n_train = len(x_train)
n_test = len(x_test)

train_dataset = tf.convert_to_tensor(x_train, dtype=tf.float32)
train_dataset = tf.data.Dataset.from_tensor_slices(train_dataset)
train_dataset = train_dataset.map(split_tf)
train_dataset = train_dataset.shuffle(n_train)
train_dataset = train_dataset.batch(mini_batch_size)

test_dataset = tf.convert_to_tensor(x_test, dtype=tf.float32)
test_dataset = tf.data.Dataset.from_tensor_slices(test_dataset)
test_dataset = test_dataset.map(split_tf)
test_dataset = test_dataset.batch(1)


def get_diastolic_end_points(data, threshold=0.1):
    end_points = []
    c = 0
    in_loop = True
    while c < len(data):
        if not in_loop and data[c] >= threshold:
            end_points.append(c)
            in_loop = True
            c += 24
        elif in_loop and data[c] < threshold:
            in_loop = False
        c += 1
    return end_points


def get_diastolic_end_points_diff(data, threshold=1):
    dt = 1
    dP = data / dt
    ddP = np.diff(dP) / dt
    ddP_peaks, _ = find_peaks(ddP)
    threshold_points = get_diastolic_end_points(data)
    c = 0
    end_points = []

    for a in threshold_points[1:]:
        closest = float('inf')
        for b in ddP_peaks:
            if abs(a - b) < closest:
                if b < a and data[b] < data[a]:
                    closest = abs(a - b)
                    point = b
        end_points.append((point))

    return ddP, end_points, threshold_points, ddP_peaks, savgol_filter(data, 51, 3)


def get_systolic_end_points(lvp, lvt, edv_points):
    E = lvp / lvt
    # E = savgol_filter(E, 51, 3)
    dt = 1
    dE = np.diff(E) / dt
    peak_thresh = (min(-dE) + max(-dE)) / 2
    dE_peaks, _ = find_peaks(-dE, prominence=0.005)
    filter_peaks = []
    for peak in dE_peaks:
        if -dE[peak] > peak_thresh:
            filter_peaks.append(peak)
    dE_peaks = filter_peaks

    esv_points = []
    for i in zip(edv_points, edv_points[1:]):
        # find farthest point
        furthest_distance = float('-inf')
        furthest_point = -1
        for j in range(i[0], i[1]):
            if math.dist([lvp[i[0]], lvt[i[0]]],
                         [lvp[j], lvt[j]]) > furthest_distance:
                furthest_point = j
                furthest_distance = math.dist([lvp[i[0]], lvt[i[0]]],
                                              [lvp[j], lvt[j]])
        if furthest_point != -1:
            esv_points.append(furthest_point)

    return dE, esv_points


def downsample(filters, size, strides, apply_batchnorm=True, padding='same'):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides, padding=padding,
                               kernel_initializer=initializer, use_bias=False))
    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())
    return result


def upsample(filters, size, strides, apply_dropout=False, padding='same'):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides,
                                        padding=padding,
                                        kernel_initializer=initializer,
                                        use_bias=False))
    result.add(tf.keras.layers.BatchNormalization())
    if apply_dropout:
        result.add(tf.keras.layers.Dropout(encoder_dropout))
    result.add(tf.keras.layers.ReLU())
    return result


def Generator():
    input_one = tf.keras.layers.Input(shape=[num_prediction_vars, prediction_length, 1])
    input_two = tf.keras.layers.Input(shape=[num_prediction_vars, prediction_length, 1])
    input_three = tf.keras.layers.Input(shape=[num_prediction_vars, prediction_length, 1])
    input_four = tf.keras.layers.Input(shape=[num_prediction_vars, prediction_length, 1])
    input_five = tf.keras.layers.Input(shape=[num_prediction_vars, prediction_length, 1])
    input_six = tf.keras.layers.Input(shape=[num_prediction_vars, prediction_length, 1])
    input_seven = tf.keras.layers.Input(shape=[num_prediction_vars, prediction_length, 1])
    input_eight = tf.keras.layers.Input(shape=[num_prediction_vars, prediction_length, 1])
    input_nine = tf.keras.layers.Input(shape=[num_prediction_vars, prediction_length, 1])
    input_ten = tf.keras.layers.Input(shape=[num_prediction_vars, prediction_length, 1])
    input_eleven = tf.keras.layers.Input(shape=[num_prediction_vars, prediction_length, 1])
    input_twelve = tf.keras.layers.Input(shape=[num_prediction_vars, prediction_length, 1])
    input_thirteen = tf.keras.layers.Input(shape=[num_prediction_vars, prediction_length, 1])
    input_fourteen = tf.keras.layers.Input(shape=[num_prediction_vars, prediction_length, 1])
    input_fifteen = tf.keras.layers.Input(shape=[num_prediction_vars, prediction_length, 1])
    input_sixteen = tf.keras.layers.Input(shape=[num_prediction_vars, prediction_length, 1])
    input_seventeen = tf.keras.layers.Input(shape=[num_prediction_vars, prediction_length, 1])
    input_eighteen = tf.keras.layers.Input(shape=[num_prediction_vars, prediction_length, 1])
    input_nineteen = tf.keras.layers.Input(shape=[num_prediction_vars, prediction_length, 1])
    input_twenty = tf.keras.layers.Input(shape=[num_prediction_vars, prediction_length, 1])

    input_vad_speed = tf.keras.layers.Input(shape=[1, input_length, 1])

    inputs = tf.keras.layers.concatenate(
        [input_one, input_two, input_three, input_four, input_five, input_six, input_seven, input_eight, input_nine,
         input_ten, input_eleven, input_twelve, input_thirteen, input_fourteen, input_fifteen, input_sixteen,
         input_seventeen, input_eighteen, input_nineteen, input_twenty], axis=2)
    inputs = tf.keras.layers.concatenate([inputs, input_vad_speed], axis=1)

    down_stack = [
        downsample(filters=64, size=(1, 101), strides=(1, 1), apply_batchnorm=False, padding='valid'),
        downsample(filters=64, size=(1, 51), strides=(1, 1), apply_batchnorm=False, padding='valid'),
        # downsample(filters=64, size=(1, 25), strides=(1, 2), apply_batchnorm=False),
        downsample(filters=1, size=(1, 26), strides=(1, 1), padding='valid'),
    ]

    up_stack = [
        upsample(filters=1, size=(1, 26), strides=(1, 1), apply_dropout=True, padding='valid'),
        # upsample(filters=64, size=(1, 25), strides=(1, 2), apply_dropout=True),
        upsample(filters=64, size=(1, 51), strides=(1, 1), apply_dropout=True, padding='valid'),
        upsample(filters=64, size=(1, 101), strides=(1, 1), apply_dropout=True, padding='valid'),
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    # last = tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=(1, 37), strides=(1, 1),
    #                                       padding='valid',
    #                                       kernel_initializer=initializer,
    #                                       activation='tanh')
    last = tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 91), strides=(1, 1),
                                  padding='valid',
                                  kernel_initializer=initializer,
                                  activation='tanh')

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    # ADD vars to Bottleneck
    # flatten -> concat -> reshape
    latent_input_phase = tf.keras.layers.Input(shape=[1])
    latent_input_inter = tf.keras.layers.Input(shape=[1])
    x = tf.keras.layers.Normalization()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Concatenate()([x, latent_input_phase, latent_input_inter])
    x = tf.keras.layers.Dense(125)(x)
    x = tf.keras.layers.Reshape((5, 25, 1))(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(40)(x)
    x = tf.keras.layers.Reshape((4, 10, 1))(x)

    return tf.keras.Model(
        inputs=[input_one, input_two, input_three, input_four, input_five, input_six, input_seven, input_eight,
                input_nine, input_ten, input_eleven, input_twelve, input_thirteen, input_fourteen, input_fifteen,
                input_sixteen, input_seventeen, input_eighteen, input_nineteen, input_twenty, input_vad_speed,
                latent_input_phase,
                latent_input_inter], outputs=x)


generator = Generator()
print(generator.summary())
tf.keras.utils.plot_model(generator, show_shapes=True, dpi=64, to_file='generator.png')
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

non_gan_loss_weight = 10


def generator_loss(disc_generated_output, gen_output, target, step):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

    # l1 absolute error
    l1_loss = tf.reduce_mean(tf.abs(tf.expand_dims(target, axis=3) - gen_output))
    dtw_loss(gen_output, target, step)
    if step < train_steps * 0.4:
        dtw_loss_weight = 0
        l1_loss_weight = 1
    elif train_steps * 0.4 < step < train_steps * 0.8:
        dtw_slide_end = 0.8
        sliding_range = train_steps * 0.8 - train_steps * 0.4
        current_step_in_range = train_steps * 0.8 - step
        dtw_loss_weight = (current_step_in_range / sliding_range) * dtw_slide_end
        l1_loss_weight = 1.0 - dtw_loss_weight

    total_gen_loss = gan_loss + non_gan_loss_weight * (l1_loss_weight * l1_loss + dtw_loss_weight * dtw_loss)

    return total_gen_loss, gan_loss, l1_loss


def Discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)

    input_one = tf.keras.layers.Input(shape=[num_prediction_vars, prediction_length, 1])
    input_two = tf.keras.layers.Input(shape=[num_prediction_vars, prediction_length, 1])
    input_three = tf.keras.layers.Input(shape=[num_prediction_vars, prediction_length, 1])
    input_four = tf.keras.layers.Input(shape=[num_prediction_vars, prediction_length, 1])
    input_five = tf.keras.layers.Input(shape=[num_prediction_vars, prediction_length, 1])
    input_six = tf.keras.layers.Input(shape=[num_prediction_vars, prediction_length, 1])
    input_seven = tf.keras.layers.Input(shape=[num_prediction_vars, prediction_length, 1])
    input_eight = tf.keras.layers.Input(shape=[num_prediction_vars, prediction_length, 1])
    input_nine = tf.keras.layers.Input(shape=[num_prediction_vars, prediction_length, 1])
    input_ten = tf.keras.layers.Input(shape=[num_prediction_vars, prediction_length, 1])
    input_eleven = tf.keras.layers.Input(shape=[num_prediction_vars, prediction_length, 1])
    input_twelve = tf.keras.layers.Input(shape=[num_prediction_vars, prediction_length, 1])
    input_thirteen = tf.keras.layers.Input(shape=[num_prediction_vars, prediction_length, 1])
    input_fourteen = tf.keras.layers.Input(shape=[num_prediction_vars, prediction_length, 1])
    input_fifteen = tf.keras.layers.Input(shape=[num_prediction_vars, prediction_length, 1])
    input_sixteen = tf.keras.layers.Input(shape=[num_prediction_vars, prediction_length, 1])
    input_seventeen = tf.keras.layers.Input(shape=[num_prediction_vars, prediction_length, 1])
    input_eighteen = tf.keras.layers.Input(shape=[num_prediction_vars, prediction_length, 1])
    input_nineteen = tf.keras.layers.Input(shape=[num_prediction_vars, prediction_length, 1])
    input_twenty = tf.keras.layers.Input(shape=[num_prediction_vars, prediction_length, 1])

    tar_first = tf.keras.layers.Input(shape=[num_prediction_vars, prediction_length, 1], name='target_first')
    tar_second = tf.keras.layers.Input(shape=[num_prediction_vars, prediction_length, 1], name='target_second')
    tar_third = tf.keras.layers.Input(shape=[num_prediction_vars, prediction_length, 1], name='target_third')
    tar_fourth = tf.keras.layers.Input(shape=[num_prediction_vars, prediction_length, 1], name='target_fourth')
    tar_fifth = tf.keras.layers.Input(shape=[num_prediction_vars, prediction_length, 1], name='target_fifth')
    tar_sixth = tf.keras.layers.Input(shape=[num_prediction_vars, prediction_length, 1], name='target_sixth')
    tar_seventh = tf.keras.layers.Input(shape=[num_prediction_vars, prediction_length, 1], name='target_seventh')
    tar_eigth = tf.keras.layers.Input(shape=[num_prediction_vars, prediction_length, 1], name='target_eigth')
    tar_ninth = tf.keras.layers.Input(shape=[num_prediction_vars, prediction_length, 1], name='target_ninth')
    tar_tenth = tf.keras.layers.Input(shape=[num_prediction_vars, prediction_length, 1], name='target_tenth')
    tar_eleventh = tf.keras.layers.Input(shape=[num_prediction_vars, prediction_length, 1], name='tar_eleventh')
    tar_twelth = tf.keras.layers.Input(shape=[num_prediction_vars, prediction_length, 1], name='tar_twelth')
    tar_thirteenth = tf.keras.layers.Input(shape=[num_prediction_vars, prediction_length, 1], name='tar_thirteenth')
    tar_fourteenth = tf.keras.layers.Input(shape=[num_prediction_vars, prediction_length, 1], name='tar_fourteenth')
    tar_fifteenth = tf.keras.layers.Input(shape=[num_prediction_vars, prediction_length, 1], name='tar_fifteenth')
    tar_sixteenth = tf.keras.layers.Input(shape=[num_prediction_vars, prediction_length, 1], name='tar_sixteenth')
    tar_seventeenth = tf.keras.layers.Input(shape=[num_prediction_vars, prediction_length, 1], name='tar_seventeenth')
    tar_eigthteenth = tf.keras.layers.Input(shape=[num_prediction_vars, prediction_length, 1], name='tar_eigthteenth')
    tar_nineteenth = tf.keras.layers.Input(shape=[num_prediction_vars, prediction_length, 1], name='tar_nineteenth')
    tar_twentieth = tf.keras.layers.Input(shape=[num_prediction_vars, prediction_length, 1], name='tar_twentieth')

    x = tf.keras.layers.concatenate(
        [input_one, input_two, input_three, input_four, input_five, input_six, input_seven, input_eight, input_nine,
         input_ten, input_eleven, input_twelve, input_thirteen, input_fourteen, input_fifteen, input_sixteen,
         input_seventeen, input_eighteen, input_nineteen, input_twenty, tar_first, tar_second, tar_third, tar_fourth,
         tar_fifth, tar_sixth, tar_seventh, tar_eigth, tar_ninth, tar_tenth, tar_eleventh, tar_twelth, tar_thirteenth,
         tar_fourteenth, tar_fifteenth, tar_sixteenth, tar_seventeenth, tar_eigthteenth, tar_nineteenth, tar_twentieth],
        axis=2)

    down1 = downsample(filters=64, size=(1, 75), strides=(1, 2), apply_batchnorm=False)(x)
    down2 = downsample(filters=128, size=(1, 50), strides=(1, 5))(down1)

    batchnorm1 = tf.keras.layers.BatchNormalization()(down2)

    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)

    last = tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 5), strides=(1, 1),
                                  kernel_initializer=initializer)(zero_pad2)

    return tf.keras.Model(
        inputs=[input_one, input_two, input_three, input_four, input_five, input_six, input_seven, input_eight,
                input_nine, input_ten, input_eleven, input_twelve, input_thirteen, input_fourteen, input_fifteen,
                input_sixteen, input_seventeen, input_eighteen, input_nineteen, input_twenty, tar_first, tar_second,
                tar_third, tar_fourth, tar_fifth, tar_sixth, tar_seventh, tar_eigth, tar_ninth, tar_tenth, tar_eleventh,
                tar_twelth, tar_thirteenth, tar_fourteenth, tar_fifteenth, tar_sixteenth, tar_seventeenth,
                tar_eigthteenth, tar_nineteenth, tar_twentieth], outputs=last)


discriminator = Discriminator()
print(discriminator.summary())
tf.keras.utils.plot_model(discriminator, show_shapes=True, dpi=64, to_file='discriminator.png')


def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss


generator_optimizer = tf.keras.optimizers.Adam(learning_rate=gen_learning_rate, beta_1=0.5)
generator_optimizer = tf.keras.mixed_precision.LossScaleOptimizer(generator_optimizer)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=disc_learning_rate, beta_1=0.5)
discriminator_optimizer = tf.keras.mixed_precision.LossScaleOptimizer(discriminator_optimizer)

checkpoint_dir = './'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)


def generate_images(model, test_input, tar, stop=False):
    prediction = model([test_input], training=False)

    num_vars = np.shape(prediction)[1]
    fig, axs = plt.subplots(num_vars + 1)
    fig.tight_layout()

    for i in range(num_vars):
        splitted_target = tf.split(tar, [1] * num_vars, axis=1)
        splitted_prediction = tf.split(prediction, [1] * num_vars, axis=1)
        l1_loss = tf.reduce_mean(tf.abs(splitted_target[i] - splitted_prediction[i]))
        axs[i].set_title("l1 error: " + str(l1_loss.numpy()))

    # plot pv curves
    curve = tar[0][1].numpy()
    curve = np.squeeze(curve)
    c = 0
    starts_low = (curve[0] < -0.8)
    start_point = 0
    for i in curve:
        if not starts_low:
            if i < -0.8:
                starts_low = True
                start_point = c
            else:
                c += 1
                continue
        if i > 0.0:
            break
        c += 1
    k1 = KneeLocator(range(len(curve[start_point:c])), curve[start_point:c], curve="convex")
    stroke_start = 0
    stroke_end = 0
    if k1.elbow is not None:
        stroke_start = k1.elbow + start_point
        stroke_start = stroke_start - 5 if stroke_start - 5 >= 0 else 0

        c_two = c
        for i in range(c_two, len(curve)):
            if curve[i] < -0.9:
                break
            c_two += 1

        k2 = KneeLocator(range(len(curve[c_two - 5:c_two + 5])), curve[c_two - 5:c_two + 5], curve="convex",
                         direction="decreasing")
        if k2.elbow is not None:
            stroke_end = k2.elbow + c_two - 5
            stroke_end = stroke_end + 5 if stroke_end + 5 < len(curve) else len(curve) - 1

            axs[-1].plot(tar[0][0][stroke_start:stroke_end], tar[0][1][stroke_start:stroke_end], color='b')
            axs[-1].plot(prediction[0][0][stroke_start:stroke_end], prediction[0][1][stroke_start:stroke_end],
                         color='y')

    # plot data
    for i in range(num_vars):
        plot_target = tf.concat(
            [test_input[0][i][:90], np.mean([test_input[0][i][90:], tar[0][i][:10]], axis=0), tar[0][i][10:]], axis=0)
        plot_prediction = tf.concat(
            [test_input[0][i][:90], np.mean([test_input[0][i][90:], prediction[0][i][:10]], axis=0),
             prediction[0][i][10:]], axis=0)
        axs[i].plot(plot_target, color='b')
        axs[i].plot(plot_prediction, color='y')
        axs[i].axvline(x=95 + stroke_start, color='k')
        axs[i].axvline(x=95 + stroke_end, color='k')
        # if i < num_vars-2:
        axs[i].set_ylim(-1, 1)
        # else:
        #    axs[i].set_ylim(0, 10)

    plt.savefig('../Data/Figures/pix2pix/' + str(datetime.datetime.now()) + '.png')


log_dir = "logs/"

summary_writer = tf.summary.create_file_writer(
    log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))


def make_long_prediction(input_one, input_two, input_three, input_four, input_five, input_six,
                         input_seven, input_eight,
                         input_nine, input_ten, input_eleven, input_twelve, input_thirteen,
                         input_fourteen, input_fifteen,
                         input_sixteen, input_seventeen, input_eighteen, input_nineteen,
                         input_twenty, speed_one, speed_two, speed_three, speed_four, speed_five, speed_six,
                         speed_seven, speed_eight,
                         speed_nine, speed_ten, speed_eleven, speed_twelve, speed_thirteen, speed_fourteen,
                         speed_fifteen,
                         speed_sixteen, speed_seventeen, speed_eighteen, speed_nineteen, speed_twenty, phase,
                         intervention,
                         eval_length=input_length + disc_future * prediction_length):
    latent_phase = phase
    latent_inter = intervention
    outputs = []
    full_sequence = tf.concat(
        [input_one, input_two, input_three, input_four, input_five, input_six, input_seven, input_eight, input_nine,
         input_ten, input_eleven, input_twelve, input_thirteen,
         input_fourteen, input_fifteen,
         input_sixteen, input_seventeen, input_eighteen, input_nineteen,
         input_twenty], axis=2)
    # full_sequence = tf.expand_dims(full_sequence, axis=0)
    c = 0
    speed_list = [speed_one, speed_two, speed_three, speed_four, speed_five, speed_six,
                  speed_seven, speed_eight,
                  speed_nine, speed_ten, speed_eleven, speed_twelve, speed_thirteen, speed_fourteen,
                  speed_fifteen,
                  speed_sixteen, speed_seventeen, speed_eighteen, speed_nineteen, speed_twenty]
    for _ in range(int((eval_length - input_length) / prediction_length)):
        next_step = generator(
            [input_one, input_two, input_three, input_four, input_five, input_six, input_seven, input_eight, input_nine,
             input_ten, input_eleven, input_twelve, input_thirteen,
             input_fourteen, input_fifteen,
             input_sixteen, input_seventeen, input_eighteen, input_nineteen,
             input_twenty, speed_list[c],
             latent_phase, latent_inter],
            training=True)
        full_sequence = tf.concat([full_sequence, tf.squeeze(next_step, axis=3)], axis=2)
        input_one = input_two
        input_two = input_three
        input_three = input_four
        input_four = input_five
        input_five = input_six
        input_six = input_seven
        input_seven = input_eight
        input_eight = input_nine
        input_nine = input_ten
        input_ten = input_eleven
        input_eleven = input_twelve
        input_twelve = input_thirteen
        input_thirteen = input_fourteen
        input_fourteen = input_fifteen
        input_fifteen = input_sixteen
        input_sixteen = input_seventeen
        input_seventeen = input_eighteen
        input_eighteen = input_nineteen
        input_nineteen = input_twenty
        input_twenty = next_step
        c += 1
    outputs.append(tf.squeeze(full_sequence, axis=0))
    return tf.convert_to_tensor(outputs)


def make_long_predictions(input_one, input_two, input_three, input_four, input_five, input_six, input_seven,
                          input_eight,
                          input_nine, input_ten, intervention, phase,
                          eval_length=input_length + disc_future * prediction_length):
    latent_var = intervention  # if phase[0] != 1 else np.float32(np.array([0.]))
    outputs = []
    full_sequence = tf.concat(
        [input_one, input_two, input_three, input_four, input_five, input_six, input_seven, input_eight, input_nine,
         input_ten], axis=2)
    next_steps = []
    for _ in range(int((eval_length - input_length) / prediction_length)):
        next_step = generator(
            [input_one, input_two, input_three, input_four, input_five, input_six, input_seven, input_eight, input_nine,
             input_ten, latent_var],
            training=True)
        next_steps.append(next_step)
        full_sequence = tf.concat([full_sequence, next_step], axis=2)
        input_one = input_two
        input_two = input_three
        input_three = input_four
        input_four = input_five
        input_five = input_six
        input_six = input_seven
        input_seven = input_eight
        input_eight = input_nine
        input_nine = input_ten
        input_ten = next_step
    outputs.append(tf.squeeze(full_sequence, axis=0))
    return tf.convert_to_tensor(outputs)


@tf.function
def train_step(batch, step, long_prediction, train_gen=True):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(
            [batch[0], batch[1], batch[2], batch[3], batch[4], batch[5], batch[6], batch[7], batch[8], batch[9],
             batch[10], batch[11], batch[12], batch[13], batch[14], batch[15], batch[16], batch[17], batch[18],
             batch[19], batch[40], batch[-2], batch[-1]],
            training=train_gen)
        gen_disc_output_first = gen_output
        gen_disc_output_second = generator(
            [batch[1], batch[2], batch[3], batch[4], batch[5], batch[6], batch[7], batch[8], batch[9],
             batch[10], batch[11], batch[12], batch[13], batch[14], batch[15], batch[16], batch[17], batch[18],
             batch[19],
             gen_disc_output_first, batch[41], batch[-2], batch[-1]], training=train_gen)
        gen_disc_output_third = generator(
            [batch[2], batch[3], batch[4], batch[5], batch[6], batch[7], batch[8], batch[9],
             batch[10], batch[11], batch[12], batch[13], batch[14], batch[15], batch[16], batch[17], batch[18],
             batch[19],
             gen_disc_output_first, gen_disc_output_second, batch[42], batch[-2], batch[-1]],
            training=train_gen)
        gen_disc_output_fourth = generator(
            [batch[3], batch[4], batch[5], batch[6], batch[7], batch[8], batch[9], batch[10], batch[11], batch[12],
             batch[13], batch[14], batch[15], batch[16], batch[17], batch[18],
             batch[19], gen_disc_output_first,
             gen_disc_output_second, gen_disc_output_third, batch[43], batch[-2], batch[-1]],
            training=train_gen)
        gen_disc_output_fifth = generator(
            [batch[4], batch[5], batch[6], batch[7], batch[8], batch[9], batch[10], batch[11], batch[12], batch[13],
             batch[14], batch[15], batch[16], batch[17], batch[18],
             batch[19], gen_disc_output_first, gen_disc_output_second,
             gen_disc_output_third, gen_disc_output_fourth, batch[44], batch[-2], batch[-1]],
            training=train_gen)
        gen_disc_output_sixth = generator(
            [batch[5], batch[6], batch[7], batch[8], batch[9], batch[10], batch[11], batch[12], batch[13], batch[14],
             batch[15], batch[16], batch[17], batch[18],
             batch[19], gen_disc_output_first, gen_disc_output_second,
             gen_disc_output_third, gen_disc_output_fourth, gen_disc_output_fifth, batch[45], batch[-2],
             batch[-1]],
            training=train_gen)
        gen_disc_output_seventh = generator(
            [batch[6], batch[7], batch[8], batch[9], batch[10], batch[11], batch[12], batch[13], batch[14], batch[15],
             batch[16], batch[17], batch[18],
             batch[19], gen_disc_output_first, gen_disc_output_second,
             gen_disc_output_third, gen_disc_output_fourth, gen_disc_output_fifth, gen_disc_output_sixth,
             batch[46], batch[-2],
             batch[-1]],
            training=train_gen)
        gen_disc_output_eigth = generator(
            [batch[7], batch[8], batch[9], batch[10], batch[11], batch[12], batch[13], batch[14], batch[15], batch[16],
             batch[17], batch[18],
             batch[19], gen_disc_output_first, gen_disc_output_second,
             gen_disc_output_third, gen_disc_output_fourth, gen_disc_output_fifth, gen_disc_output_sixth,
             gen_disc_output_seventh, batch[47], batch[-2], batch[-1]],
            training=train_gen)
        gen_disc_output_ninth = generator(
            [batch[8], batch[9], batch[10], batch[11], batch[12], batch[13], batch[14], batch[15], batch[16], batch[17],
             batch[18],
             batch[19], gen_disc_output_first, gen_disc_output_second,
             gen_disc_output_third, gen_disc_output_fourth, gen_disc_output_fifth, gen_disc_output_sixth,
             gen_disc_output_seventh, gen_disc_output_eigth, batch[48], batch[-2], batch[-1]],
            training=train_gen)
        gen_disc_output_tenth = generator(
            [batch[9], batch[10], batch[11], batch[12], batch[13], batch[14], batch[15], batch[16], batch[17],
             batch[18],
             batch[19], gen_disc_output_first, gen_disc_output_second,
             gen_disc_output_third, gen_disc_output_fourth, gen_disc_output_fifth, gen_disc_output_sixth,
             gen_disc_output_seventh, gen_disc_output_eigth, gen_disc_output_ninth, batch[49], batch[-2],
             batch[-1]],
            training=train_gen)
        gen_disc_output_eleventh = generator(
            [batch[10], batch[11], batch[12], batch[13], batch[14], batch[15], batch[16], batch[17], batch[18],
             batch[19], gen_disc_output_first, gen_disc_output_second,
             gen_disc_output_third, gen_disc_output_fourth, gen_disc_output_fifth, gen_disc_output_sixth,
             gen_disc_output_seventh, gen_disc_output_eigth, gen_disc_output_ninth, gen_disc_output_tenth,
             batch[50], batch[-2],
             batch[-1]],
            training=train_gen)
        gen_disc_output_twelfth = generator(
            [batch[11], batch[12], batch[13], batch[14], batch[15], batch[16], batch[17], batch[18],
             batch[19], gen_disc_output_first, gen_disc_output_second,
             gen_disc_output_third, gen_disc_output_fourth, gen_disc_output_fifth, gen_disc_output_sixth,
             gen_disc_output_seventh, gen_disc_output_eigth, gen_disc_output_ninth, gen_disc_output_tenth,
             gen_disc_output_eleventh, batch[51], batch[-2], batch[-1]],
            training=train_gen)
        gen_disc_output_thirteenth = generator(
            [batch[12], batch[13], batch[14], batch[15], batch[16], batch[17], batch[18],
             batch[19], gen_disc_output_first, gen_disc_output_second, gen_disc_output_third, gen_disc_output_fourth,
             gen_disc_output_fifth, gen_disc_output_sixth,
             gen_disc_output_seventh, gen_disc_output_eigth, gen_disc_output_ninth, gen_disc_output_tenth,
             gen_disc_output_eleventh, gen_disc_output_twelfth, batch[52], batch[-2], batch[-1]],
            training=train_gen)
        gen_disc_output_fourteenth = generator(
            [batch[13], batch[14], batch[15], batch[16], batch[17], batch[18],
             batch[19], gen_disc_output_first, gen_disc_output_second, gen_disc_output_third, gen_disc_output_fourth,
             gen_disc_output_fifth, gen_disc_output_sixth,
             gen_disc_output_seventh, gen_disc_output_eigth, gen_disc_output_ninth, gen_disc_output_tenth,
             gen_disc_output_eleventh, gen_disc_output_twelfth, gen_disc_output_thirteenth, batch[53],
             batch[-2], batch[-1]],
            training=train_gen)
        gen_disc_output_fifteenth = generator(
            [batch[14], batch[15], batch[16], batch[17], batch[18],
             batch[19], gen_disc_output_first, gen_disc_output_second, gen_disc_output_third, gen_disc_output_fourth,
             gen_disc_output_fifth, gen_disc_output_sixth,
             gen_disc_output_seventh, gen_disc_output_eigth, gen_disc_output_ninth, gen_disc_output_tenth,
             gen_disc_output_eleventh, gen_disc_output_twelfth, gen_disc_output_thirteenth, gen_disc_output_fourteenth,
             batch[54],
             batch[-2], batch[-1]],
            training=train_gen)
        gen_disc_output_sixteenth = generator(
            [batch[15], batch[16], batch[17], batch[18],
             batch[19], gen_disc_output_first, gen_disc_output_second, gen_disc_output_third, gen_disc_output_fourth,
             gen_disc_output_fifth, gen_disc_output_sixth,
             gen_disc_output_seventh, gen_disc_output_eigth, gen_disc_output_ninth, gen_disc_output_tenth,
             gen_disc_output_eleventh, gen_disc_output_twelfth, gen_disc_output_thirteenth, gen_disc_output_fourteenth,
             gen_disc_output_fifteenth, batch[55], batch[-2], batch[-1]],
            training=train_gen)
        gen_disc_output_seventeenth = generator(
            [batch[16], batch[17], batch[18],
             batch[19], gen_disc_output_first, gen_disc_output_second, gen_disc_output_third, gen_disc_output_fourth,
             gen_disc_output_fifth, gen_disc_output_sixth, gen_disc_output_seventh, gen_disc_output_eigth,
             gen_disc_output_ninth, gen_disc_output_tenth,
             gen_disc_output_eleventh, gen_disc_output_twelfth, gen_disc_output_thirteenth, gen_disc_output_fourteenth,
             gen_disc_output_fifteenth,
             gen_disc_output_sixteenth, batch[56], batch[-2], batch[-1]],
            training=train_gen)
        gen_disc_output_eighteenth = generator(
            [batch[17], batch[18],
             batch[19], gen_disc_output_first, gen_disc_output_second, gen_disc_output_third, gen_disc_output_fourth,
             gen_disc_output_fifth, gen_disc_output_sixth, gen_disc_output_seventh, gen_disc_output_eigth,
             gen_disc_output_ninth, gen_disc_output_tenth,
             gen_disc_output_eleventh, gen_disc_output_twelfth, gen_disc_output_thirteenth, gen_disc_output_fourteenth,
             gen_disc_output_fifteenth,
             gen_disc_output_sixteenth, gen_disc_output_seventeenth, batch[57], batch[-2], batch[-1]],
            training=train_gen)
        gen_disc_output_nineteenth = generator(
            [batch[18],
             batch[19], gen_disc_output_first, gen_disc_output_second, gen_disc_output_third, gen_disc_output_fourth,
             gen_disc_output_fifth, gen_disc_output_sixth, gen_disc_output_seventh, gen_disc_output_eigth,
             gen_disc_output_ninth, gen_disc_output_tenth,
             gen_disc_output_eleventh, gen_disc_output_twelfth, gen_disc_output_thirteenth, gen_disc_output_fourteenth,
             gen_disc_output_fifteenth,
             gen_disc_output_sixteenth, gen_disc_output_seventeenth, gen_disc_output_eighteenth, batch[58],
             batch[-2], batch[-1]],
            training=train_gen)
        gen_disc_output_twentieth = generator(
            [batch[19], gen_disc_output_first, gen_disc_output_second, gen_disc_output_third, gen_disc_output_fourth,
             gen_disc_output_fifth, gen_disc_output_sixth, gen_disc_output_seventh, gen_disc_output_eigth,
             gen_disc_output_ninth, gen_disc_output_tenth,
             gen_disc_output_eleventh, gen_disc_output_twelfth, gen_disc_output_thirteenth, gen_disc_output_fourteenth,
             gen_disc_output_fifteenth, gen_disc_output_sixteenth, gen_disc_output_seventeenth,
             gen_disc_output_eighteenth,
             gen_disc_output_nineteenth, batch[59], batch[-2], batch[-1]],
            training=train_gen)

        disc_real_output = discriminator(
            [batch[0], batch[1], batch[2], batch[3], batch[4], batch[5], batch[6], batch[7], batch[8], batch[9],
             batch[10], batch[11], batch[12], batch[13], batch[14], batch[15], batch[16], batch[17], batch[18],
             batch[19], batch[20], batch[21], batch[22], batch[23], batch[24], batch[25], batch[26], batch[27],
             batch[28], batch[29], batch[30], batch[31], batch[32], batch[33], batch[34], batch[35], batch[36],
             batch[37],
             batch[38], batch[39]], training=True)
        disc_generated_output = discriminator(
            [batch[0], batch[1], batch[2], batch[3], batch[4], batch[5], batch[6], batch[7], batch[8], batch[9],
             batch[10], batch[11], batch[12], batch[13], batch[14], batch[15], batch[16], batch[17], batch[18],
             batch[19],
             gen_disc_output_first, gen_disc_output_second,
             gen_disc_output_third, gen_disc_output_fourth,
             gen_disc_output_fifth, gen_disc_output_sixth, gen_disc_output_seventh, gen_disc_output_eigth,
             gen_disc_output_ninth, gen_disc_output_tenth, gen_disc_output_eleventh, gen_disc_output_twelfth,
             gen_disc_output_thirteenth, gen_disc_output_fourteenth, gen_disc_output_fifteenth,
             gen_disc_output_sixteenth, gen_disc_output_seventeenth, gen_disc_output_eighteenth,
             gen_disc_output_nineteenth, gen_disc_output_twentieth], training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output,
                                                                   batch[20], step)
        scaled_gen_loss = generator_optimizer.get_scaled_loss(gen_total_loss)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)
        scaled_disc_loss = discriminator_optimizer.get_scaled_loss(disc_loss)

    scaled_generator_gradients = gen_tape.gradient(scaled_gen_loss,
                                                   generator.trainable_variables)
    scaled_discriminator_gradients = disc_tape.gradient(scaled_disc_loss,
                                                        discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(scaled_generator_gradients,
                                            generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(scaled_discriminator_gradients,
                                                discriminator.trainable_variables))

    with summary_writer.as_default():
        tf.summary.scalar('gen_total_loss', gen_total_loss, step=step)
        tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=step)
        tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=step)
        tf.summary.scalar('disc_loss', disc_loss, step=step)

    return gen_total_loss, gen_gan_loss, gen_l1_loss


def evaluate(test_dataset, step, full=False):
    l1_loss_one = 0
    l1_loss_two = 0
    l1_loss_three = 0
    l1_loss_four = 0
    c = 0
    eval_dir = result_dir + "eval_" + str(step.numpy()) + "_" + str(full)
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)
    for (
            input_one, input_two, input_three, input_four, input_five, input_six, input_seven, input_eight,
            input_nine, input_ten, input_eleven, input_twelve, input_thirteen, input_fourteen, input_fifteen,
            input_sixteen, input_seventeen, input_eighteen, input_nineteen, input_twenty,
            _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
            speed_one, speed_two, speed_three, speed_four, speed_five, speed_six, speed_seven, speed_eight,
            speed_nine, speed_ten, speed_eleven, speed_twelve, speed_thirteen, speed_fourteen, speed_fifteen,
            speed_sixteen, speed_seventeen, speed_eighteen, speed_nineteen, speed_twenty,
            target_sequence_disc, target_sequence_full, phase,
            intervention) in test_dataset.as_numpy_iterator():
        if full:
            prediction = make_long_prediction(input_one, input_two, input_three, input_four, input_five, input_six,
                                              input_seven, input_eight,
                                              input_nine, input_ten, input_eleven, input_twelve, input_thirteen,
                                              input_fourteen, input_fifteen,
                                              input_sixteen, input_seventeen, input_eighteen, input_nineteen,
                                              input_twenty, speed_one, speed_two, speed_three, speed_four, speed_five,
                                              speed_six, speed_seven, speed_eight,
                                              speed_nine, speed_ten, speed_eleven, speed_twelve, speed_thirteen,
                                              speed_fourteen, speed_fifteen,
                                              speed_sixteen, speed_seventeen, speed_eighteen, speed_nineteen,
                                              speed_twenty, phase, intervention,
                                              eval_length=data_window_length)
            splitted_target = tf.split(target_sequence_full, [1, 1, 1, 1], axis=1)
        else:
            prediction = make_long_prediction(input_one, input_two, input_three, input_four, input_five, input_six,
                                              input_seven, input_eight,
                                              input_nine, input_ten, input_eleven, input_twelve, input_thirteen,
                                              input_fourteen, input_fifteen,
                                              input_sixteen, input_seventeen, input_eighteen, input_nineteen,
                                              input_twenty, speed_one, speed_two, speed_three, speed_four, speed_five,
                                              speed_six, speed_seven, speed_eight,
                                              speed_nine, speed_ten, speed_eleven, speed_twelve, speed_thirteen,
                                              speed_fourteen, speed_fifteen,
                                              speed_sixteen, speed_seventeen, speed_eighteen, speed_nineteen,
                                              speed_twenty, phase, intervention)
            splitted_target = tf.split(target_sequence_disc, [1, 1, 1, 1], axis=1)
        if tf.reduce_any(tf.math.is_nan(prediction)):
            print("prediction contains nan")
        splitted_prediction = tf.split(prediction, [1, 1, 1, 1], axis=1)
        l1_loss_one_temp = tf.reduce_mean(tf.abs(splitted_target[0] - splitted_prediction[0]))
        l1_loss_two_temp = tf.reduce_mean(tf.abs(splitted_target[1] - splitted_prediction[1]))
        l1_loss_three_temp = tf.reduce_mean(tf.abs(splitted_target[2] - splitted_prediction[2]))
        l1_loss_four_temp = tf.reduce_mean(tf.abs(splitted_target[3] - splitted_prediction[3]))
        l1_loss_one += l1_loss_one_temp
        l1_loss_two += l1_loss_two_temp
        l1_loss_three += l1_loss_three_temp
        l1_loss_four += l1_loss_four_temp

        _, axs = plt.subplots(num_prediction_vars)
        for i in range(num_prediction_vars):
            axs[i].plot(splitted_target[i][0][0], color='b')
            axs[i].plot(splitted_prediction[i][0][0], color='y')
            axs[i].set_ylim(-1, 1)

        plt.savefig(eval_dir + "/" + str(datetime.datetime.now()) + '.png')
        plt.close()

        c += 1
    with open(eval_dir + "/results.txt", "w") as result_file:
        result_file.write("L1 loss var 1: " + str(l1_loss_one / c) + "\nL1 loss var 2: " + str(
            l1_loss_two / c) + "\nL1 loss var 3: " + str(l1_loss_three / c) + "\nL1 loss var 4: " + str(
            l1_loss_four / c))
    return l1_loss_one / c, l1_loss_two / c, l1_loss_three / c, l1_loss_four / c


def evaluate_single(input_image, target_image):
    prediction = generator([input_image], training=False)
    if tf.reduce_any(tf.math.is_nan(prediction)):
        print("prediction contains nan")
    splitted_target = tf.split(target_image, [1, 1], axis=1)
    splitted_prediction = tf.split(prediction, [1, 1], axis=1)
    l1_loss_one = tf.reduce_mean(tf.abs(splitted_target[0] - splitted_prediction[0]))
    l1_loss_two = tf.reduce_mean(tf.abs(splitted_target[1] - splitted_prediction[1]))

    return l1_loss_one, l1_loss_two


def evaluate_disc(data_set):
    acc = 0
    c = 0
    for (input_image, target_image, target_sequence) in data_set.as_numpy_iterator():
        # create long prediction window for disc
        long_prediction = make_long_prediction(input_image)

        disc_real = discriminator([input_image, target_sequence], training=True)
        disc_gen = discriminator([input_image, long_prediction], training=True)

        acc += tf.reduce_sum(tf.where(disc_real >= 0.5, tf.ones_like(disc_real), tf.zeros_like(disc_real))).numpy()
        acc += tf.reduce_sum(tf.where(disc_gen < 0.5, tf.ones_like(disc_gen), tf.zeros_like(disc_gen))).numpy()
        c += 64
    return acc / c


def fit(train_ds, test_ds, steps, train_gen=True):
    start = time.time()
    l1_losses_one = []
    l1_losses_two = []
    l1_losses_three = []
    l1_losses_four = []

    for step, batch in train_ds.repeat().take(steps).enumerate():
        # for step, sample in enumerate(full_batch):
        if (step) % 1000 == 0:
            display.clear_output(wait=True)

            if step != 0:
                print(f'Time taken for 1000 steps: {time.time() - start:.2f} sec\n')

            start = time.time()
            test_ds = test_ds.shuffle(n_test)
            l1_loss_one, l1_loss_two, l1_loss_three, l1_loss_four = evaluate(test_dataset.take(20), step)
            l1_losses_one.append(l1_loss_one)
            l1_losses_two.append(l1_loss_two)
            l1_losses_three.append(l1_loss_three)
            l1_losses_four.append(l1_loss_four)
            print(f"Step: {step // 1000}k")

        train_step(batch, step, train_gen)

        # Training step
        if (step + 1) % 10 == 0:
            print('.', end='', flush=True)

        # Save (checkpoint) the model every 5k steps
        if (step + 1) % 5000 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)


fit(train_dataset, test_dataset, steps=train_steps)
generator.save(result_dir + "generator.keras")
