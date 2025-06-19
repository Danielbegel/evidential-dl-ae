from tensorflow.keras import Model, layers
import tensorflow as tf

class AeModel(Model):
    def __init__(self, train_config):
        super(AeModel, self).__init__()
        self.latent_dim = train_config["encoder_design"]["latent_layer_dimension"]
        self.input_shape = (train_config["encoder_design"]["input_layer_dimension"],)

        # Encoder
        self.encoder_input = layers.Input(shape=self.input_shape)
        x = self.encoder_input
        ip_layer_dimension_list = train_config["encoder_design"]["inner_layer_dimensions"]
        for dimension in ip_layer_dimension_list:
            x = layers.Dense(dimension, activation='relu', name=f"encoder_{dimension}")(x)
        self.z = layers.Dense(self.latent_dim, activation='relu', name="latent")(x)

        # Decoder
        self.decoder_input = layers.Input(shape=(self.latent_dim,))
        x = self.decoder_input
        op_layer_dimension_list = train_config["decoder_design"]["inner_layer_dimensions"]
        for dimension in op_layer_dimension_list:
            x = layers.Dense(dimension, activation='relu')(x)
        self.decoder_output = layers.Dense(self.input_shape[0], activation='sigmoid')(x)

        # Combine encoder and decoder
        self.encoder = Model(self.encoder_input, self.z, name="encoder")
        self.encoder.summary()
        self.decoder = Model(self.decoder_input, self.decoder_output, name="decoder")
        self.decoder.summary()

        # Full autoencoder model
        full_input = self.encoder_input
        encoded = self.encoder(full_input)
        decoded = self.decoder(encoded)
        self.autoencoder = Model(full_input, decoded, name="autoencoder")

    # default: model.compile(optimizer='adam', loss='mse')
    def call(self, inputs):
        return self.autoencoder(inputs)

