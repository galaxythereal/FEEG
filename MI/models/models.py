
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, DepthwiseConv2D, Activation,
    MaxPooling2D, SeparableConv2D, Flatten, Dense, Dropout,
    SpatialDropout2D, GlobalAveragePooling2D, AveragePooling2D, Multiply
)
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.regularizers import l2

def FEEGNet(nb_classes, Chans=64, Samples=128,
           dropoutRate=0.5, kernLength=64, F1=8,
           D=2, F2=16, norm_rate=0.25, dropoutType='Dropout'):
    """
    Keras Implementation of FEEGNet (formerly EEGNet)
    Based on: http://iopscience.iop.org/article/10.1088/1741-2552/aa7808
    """

    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D or Dropout')

    input_layer = Input(shape=(Samples, Chans, 1))

    # Layer 1: Temporal Convolution
    block1 = Conv2D(F1, (kernLength, 1), padding='same',
                    input_shape=(Samples, Chans, 1),
                    use_bias=False)(input_layer)
    block1 = BatchNormalization(axis=-1)(block1)

    # Layer 2: Depthwise Convolution
    block1 = DepthwiseConv2D((1, Chans), use_bias=False,
                             depth_multiplier=D,
                             depthwise_constraint=max_norm(norm_rate))(block1)
    block1 = BatchNormalization(axis=-1)(block1)
    block1 = Activation('elu')(block1)
    block1 = dropoutType(dropoutRate)(block1)
    block1 = MaxPooling2D((4, 1))(block1)

    # Layer 3: Separable Convolution
    block2 = SeparableConv2D(F2, (16, 1),
                             use_bias=False, padding='same')(block1)
    block2 = BatchNormalization(axis=-1)(block2)
    block2 = Activation('elu')(block2)
    block2 = dropoutType(dropoutRate)(block2)
    block2 = MaxPooling2D((8, 1))(block2)

    # Layer 4: Classification
    flatten = Flatten(name='flatten')(block2)

    dense = Dense(nb_classes, name='dense',
                  kernel_constraint=max_norm(norm_rate))(flatten)
    softmax = Activation('softmax', name='softmax')(dense)

    return Model(inputs=input_layer, outputs=softmax)


def FEEGNet_Attention(nb_classes, Chans, Samples,
                    dropoutRate=0.5, kernLength=64, F1=8,
                    D=2, F2=16, norm_rate=0.25):
    """FEEGNet with Squeeze-and-Excitation attention blocks"""
    input = Input(shape=(Samples, Chans, 1))

    # Block 1
    x = Conv2D(F1, (kernLength, 1), padding='same', use_bias=False)(input)
    x = BatchNormalization(axis=-1)(x)
    x = DepthwiseConv2D((1, Chans), use_bias=False, depth_multiplier=D,
                       depthwise_constraint=max_norm(norm_rate))(x)
    x = BatchNormalization(axis=-1)(x)

    # SE Attention Block
    se = GlobalAveragePooling2D()(x)
    se = Dense(max(1, F1*D//8), activation='relu')(se)
    se = Dense(F1*D, activation='sigmoid')(se)
    x = Multiply()([x, se])

    x = Activation('elu')(x)
    x = AveragePooling2D((4, 1))(x)
    x = Dropout(dropoutRate)(x)

    # Block 2
    x = SeparableConv2D(F2, (16, 1), use_bias=False, padding='same')(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('elu')(x)
    x = AveragePooling2D((8, 1))(x)
    x = Dropout(dropoutRate)(x)

    # Classification
    x = Flatten()(x)
    x = Dense(nb_classes, kernel_constraint=max_norm(norm_rate))(x)
    output = Activation('softmax')(x)

    return Model(inputs=input, outputs=output)

def FEEGNet_Attention_Improved(nb_classes, Chans, Samples,
                              dropoutRate=0.6, kernLength=64, F1=16,
                              D=2, F2=32, norm_rate=0.25):
    """Improved FEEGNet with Squeeze-and-Excitation attention blocks and minor architectural changes"""
    input = Input(shape=(Samples, Chans, 1))

    # Block 1
    x = Conv2D(F1, (kernLength, 1), padding='same', use_bias=False,
               kernel_regularizer=l2(0.001))(input) # Added L2 regularization
    x = BatchNormalization(axis=-1)(x)
    x = DepthwiseConv2D((1, Chans), use_bias=False, depth_multiplier=D,
                       depthwise_constraint=max_norm(norm_rate))(x)
    x = BatchNormalization(axis=-1)(x)

    # SE Attention Block
    se = GlobalAveragePooling2D()(x)
    se = Dense(max(1, F1*D//8), activation='relu')(se)
    se = Dense(F1*D, activation='sigmoid')(se)
    x = Multiply()([x, se])

    x = Activation('elu')(x)
    x = AveragePooling2D((4, 1))(x)
    x = Dropout(dropoutRate)(x) # Increased dropout rate

    # Block 2
    x = SeparableConv2D(F2, (16, 1), use_bias=False, padding='same',
                        pointwise_regularizer=l2(0.001))(x) # Corrected to pointwise_regularizer
    x = BatchNormalization(axis=-1)(x)
    x = Activation('elu')(x)
    x = AveragePooling2D((8, 1))(x)
    x = Dropout(dropoutRate)(x) # Increased dropout rate

    # Added Block 3 (another Separable Convolutional Block)
    x = SeparableConv2D(F2 * 2, (16, 1), use_bias=False, padding='same', # Increased filters
                        pointwise_regularizer=l2(0.001))(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('elu')(x)
    x = AveragePooling2D((2, 1))(x) # Smaller pooling to retain more information
    x = Dropout(dropoutRate)(x)

    # Classification
    x = Flatten()(x)
    x = Dense(nb_classes, kernel_constraint=max_norm(norm_rate),
              kernel_regularizer=l2(0.001))(x) # Added L2 regularization
    output = Activation('softmax')(x)

    return Model(inputs=input, outputs=output)

