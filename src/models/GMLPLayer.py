import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa


class GMLPLayer(layers.Layer):
    def __init__(
        self, num_patches: int, embedding_dim: int, dropout_rate: float, *args, **kwargs
    ):
        super(GMLPLayer, self).__init__(*args, **kwargs)

        self.channel_projection1 = keras.Sequential(
            [
                layers.Dense(units=embedding_dim * 2),
                tfa.layers.GELU(),
                layers.Dropout(rate=dropout_rate),
            ]
        )
        self.channel_projection2 = layers.Dense(units=embedding_dim)
        self.spatial_projection = layers.Dense(
            units=num_patches, bias_initializer="Ones"
        )
        self.normalize1 = layers.LayerNormalization(epsilon=1e-6)
        self.normalize2 = layers.LayerNormalization(epsilon=1e-6)

    def spatial_gating_unit(self, x):
        u, v = tf.split(x, num_or_size_splits=2, axis=2)
        v = self.normalize2(v)
        v_channels = tf.linalg.matrix_transpose(v)
        v_projected = self.spatial_projection(v_channels)
        v_projected = tf.linalg.matrix_transpose(v_projected)

        return u * v_projected

    def call(self, inputs):
        x = self.normalize1(inputs)
        x_projected = self.channel_projection1(x)
        x_spatial = self.spatial_gating_unit(x_projected)
        x_projected = self.channel_projection2(x_spatial)

        return x + x_projected
