import torch
from UNet import Encoder, Decoder


def test_encoder_valid_shape():
    encoder = Encoder(num_input_channels=1, base_channel_size=32, latent_dim=256)
    x = torch.randn(10000, 1, 28, 28)
    res_shape = encoder(x).shape
    assert res_shape[0] == 10000 and res_shape[1] == 256, "Correct encoding dimensions"


def test_decoder_valid_shape():
    decoder = Decoder(num_input_channels=1, base_channel_size=32, latent_dim=256)
    x = torch.randn(1000, 256)
    res_shape = decoder(x).shape
    assert res_shape[0] == 1000 and res_shape[1] == 1, "Correct decoding dimensions"


if __name__ == "__main__":
    test_encoder_valid_shape()
    test_decoder_valid_shape()
    print("Everything passed")