import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from src.loss import PSNR_non_training as PSNR

class EDSRModel(tf.keras.Model):
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def predict_step(self, x):
        x = tf.cast(tf.expand_dims(x, axis=0), tf.float32)
        y_pred = self(x, training=False)
        y_pred = tf.clip_by_value(y_pred, 0, 255)
        y_pred = tf.round(y_pred)
        y_pred = tf.squeeze(tf.cast(y_pred, tf.uint8), axis=0)
        return y_pred

def ResBlock(inputs):
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(inputs)
    x = layers.Conv2D(64, 3, padding="same")(x)
    return layers.Add()([inputs, x])


def Upsampling(inputs, scale=2, num_filters=64, **kwargs):
    x = inputs

    if scale and (scale & (scale - 1)) == 0 and scale > 1: # 2^d
        steps = int(tf.math.log(tf.cast(scale, tf.float32)) / tf.math.log(2.0))
        for _ in range(steps):
            x = layers.Conv2D(num_filters * (2 ** 2), 3, padding="same", **kwargs)(x)
            x = layers.Lambda(lambda x: tf.nn.depth_to_space(x, 2))(x)
    else:
        raise ValueError(f"{scale} is invalid, should be 2,4,8,16 ...")
    return x


def build_model(num_filters=64, num_res_blocks=16, scale=2, metric="psnr", loss="mae"):
    inputs = layers.Input(shape=(None, None, 3))
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = x_skip = layers.Conv2D(num_filters, 3, padding="same")(x)

    for _ in range(num_res_blocks):
        x = ResBlock(x)

    x = layers.Conv2D(num_filters, 3, padding="same")(x)
    x = layers.Add()([x, x_skip])

    x = Upsampling(x, scale=scale, num_filters=num_filters)
    outputs = layers.Conv2D(3, 3, padding="same")(x)
    outputs = layers.Rescaling(255)(outputs)

    model = EDSRModel(inputs, outputs)

    metric = metric.lower()
    if metric == "psnr":
        metric = PSNR
    else:
        raise ValueError(f"Metric {metric} not recognized.")

    optim_edsr = keras.optimizers.Adam(
        learning_rate=keras.optimizers.schedules.PiecewiseConstantDecay(
            boundaries=[5000], values=[1e-4, 5e-5]
        )
    )
    model.compile(optimizer=optim_edsr, loss=loss, metrics=[metric])
    return model

def train_model(model, train_ds, val_ds, epochs, scale, metric="psnr"):
    epochs  = epochs
    weights_filename = f'weights_{metric}_{epochs}epochs_{scale}scale.weights.h5'

    history = model.fit(train_ds, epochs=epochs, steps_per_epoch=200, validation_data=val_ds)
    model.save_weights(weights_filename)
    return model, history

