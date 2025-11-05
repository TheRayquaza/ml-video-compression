import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from src.loss import SSIM, PSNR

class EDSRModel(tf.keras.Model):
    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        for metric in self.metrics:
            if metric.name != 'loss':
                metric.update_state(y, y_pred)
        
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data
        y_pred = self(x, training=False)
        
        self.compiled_loss(y, y_pred, regularization_losses=self.losses)
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
    metrics = []
    metric = metric.lower()
    if metric == "psnr":
        metrics = [PSNR]
    elif metric == "ssim":
        metrics = [SSIM]
    elif metric == "both":
        metrics = [PSNR, SSIM]
    else:
        raise ValueError(f"Metric {metric} not recognized.")

    optim_edsr = keras.optimizers.Adam(
        learning_rate=keras.optimizers.schedules.PiecewiseConstantDecay(
            boundaries=[5000], values=[1e-4, 5e-5]
        )
    )
    model.compile(optimizer=optim_edsr, loss=loss, metrics=metrics)
    return model

class SaveWeightsAtEpochs(Callback):
    def __init__(self, save_epochs, base_filename):
        super().__init__()
        self.save_epochs = set(save_epochs)
        self.base_filename = base_filename

    def on_epoch_end(self, epoch, logs=None):
        current_epoch = epoch + 1
        if current_epoch in self.save_epochs:
            filename = f"{self.base_filename}_epoch{current_epoch}.weights.h5"
            self.model.save_weights(filename)
            print(f"\nSaved weights at epoch {current_epoch} to {filename}")

def train_model(model, train_ds, val_ds, epochs, scale, saving_epochs=None):
    if saving_epochs is None:
        saving_epochs = []

    base_filename = f'weights_{epochs}epochs_{scale}scale'

    callbacks = [SaveWeightsAtEpochs(saving_epochs, base_filename)] if saving_epochs else []

    history = model.fit(
        train_ds, 
        epochs=epochs, 
        steps_per_epoch=200, 
        validation_data=val_ds,
        callbacks=callbacks
    )
    
    # Save final weights
    final_weights_filename = f"{base_filename}_final.weights.h5"
    model.save_weights(final_weights_filename)
    print(f"Saved final weights to {final_weights_filename}")
    
    return model, history
