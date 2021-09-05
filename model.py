from keras.layers import Input, Conv2D, Dense, Flatten, AveragePooling2D, Add, Concatenate
from keras.models import Model


def brief_net(input_shape=(170, 275, 3), output_shape=1):
    input = Input(shape=input_shape)
    x = Conv2D(filters=128, kernel_size=1, strides=1, padding='same', activation='relu')(input)
    shortcut_ = Conv2D(filters=32, kernel_size=3, strides=4, padding='same', activation='relu')(x)
    x = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu')(x)

    shortcut = AveragePooling2D(pool_size=2, strides=2, padding='same')(x)
    shortcut = Conv2D(filters=16, kernel_size=3, strides=2, padding="same", activation='relu')(shortcut)
    x = Conv2D(filters=32, kernel_size=5, strides=2, padding='same', activation='relu')(x)
    x = Conv2D(filters=16, kernel_size=3, strides=2, padding='same', activation='relu')(x)
    x = Add()([x, shortcut])
    x = Concatenate(axis=-1)([x, shortcut_])
    x = Conv2D(filters=8, kernel_size=3, strides=2, padding='same', activation='relu')(x)
    x = Flatten()(x)
    x = Dense(units=12, activation='relu')(x)

    x = Dense(units=6, activation='relu')(x)

    output = Dense(units=output_shape, activation="sigmoid")(x)

    model = Model(input, output)

    return model

if __name__ == '__main__':
    model = brief_net()
    model.summary()