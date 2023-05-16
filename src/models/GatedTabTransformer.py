from typing import List
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


from src.models.TransformerBlock import TransformerBlock
from src.models.GMLPLayer import GMLPLayer


class GatedTabTransformer(keras.Model):
    def __init__(
        self,
        categories: List[str],
        num_continuous: int,
        dim: int,
        dim_out: int,
        depth: int,
        embedding_dim: int,
        heads: int,
        attn_dropout: float,
        ff_dropout: float,
        gmlp_blocks: int,
        normalize_continuous=True,
    ):
        super(GatedTabTransformer, self).__init__()

        self.embedding_dim = embedding_dim
        self.normalize_continuous = normalize_continuous
        if normalize_continuous:
            self.continuous_normalization = layers.LayerNormalization()

        self.embedding_layers = []
        for number_of_classes in categories:
            self.embedding_layers.append(
                layers.Embedding(input_dim=number_of_classes, output_dim=dim)
            )

        self.embedded_concatenation = layers.Concatenate(axis=1)

        self.transformers = []
        for _ in range(depth):
            self.transformers.append(
                TransformerBlock(embed_dim=dim, num_heads=heads, ff_dim=dim)
            )
        self.flatten_transformer_output = layers.Flatten()

        self.pre_mlp_concatenation = layers.Concatenate()
        self.gmlp_layers = []
        for _ in range(gmlp_blocks):
            self.gmlp_layers.append(
                GMLPLayer(
                    num_patches=1, embedding_dim=self.embedding_dim, dropout_rate=0.2
                )
            )
        self.embedder2 = layers.Dense(self.embedding_dim)
        self.output_layer = layers.Dense(dim_out, activation="sigmoid")

    def call(self, inputs):
        continuous_inputs = inputs[0]
        categorical_inputs = inputs[1:]

        if self.normalize_continuous:
            continuous_inputs = self.continuous_normalization(continuous_inputs)

        embedding_outputs = []
        for categorical_input, embedding_layer in zip(
            categorical_inputs, self.embedding_layers
        ):
            embedding_outputs.append(embedding_layer(categorical_input))
        categorical_inputs = self.embedded_concatenation(embedding_outputs)

        for transformer in self.transformers:
            categorical_inputs = transformer(categorical_inputs)
        contextual_embedding = self.flatten_transformer_output(categorical_inputs)

        mlp_input = self.pre_mlp_concatenation(
            [continuous_inputs, contextual_embedding]
        )
        gmlp_input = tf.expand_dims(self.embedder2(mlp_input), axis=1)
        for gmlp_layer in self.gmlp_layers:
            gmlp_input = gmlp_layer(gmlp_input)
        gmlp_input = tf.math.reduce_mean(gmlp_input, axis=1)

        return self.output_layer(gmlp_input)
