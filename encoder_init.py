import sys
import torch
from autoencoder.encoder import VariationalEncoder
from parameters import COOP_SECTORS, IM_HEIGHT, IM_WIDTH

class EncodeState():
    def __init__(self, latent_dim):
        self.latent_dim = latent_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            self.conv_encoder = VariationalEncoder(self.latent_dim).to(self.device)
            self.conv_encoder.load()
            self.conv_encoder.eval()

            for params in self.conv_encoder.parameters():
                params.requires_grad = False
        except:
            print('Encoder could not be initialized.')
            sys.exit()
    
    def process(self, observation):
        image_obs = torch.tensor(observation[0], dtype=torch.float).to(self.device)
        if COOP_SECTORS:
            # Upstream fed the VAE a (W,H)-scrambled BGR 0..255 frame, which sometimes overflows exp(sigma) to inf/NaN.
            # Give it what vae.py trained on (ToTensor: RGB, 0..1, CHW). The buffer order is intact, so reshape restores the image.
            image_obs = image_obs.reshape(IM_HEIGHT, IM_WIDTH, 3)[..., [2, 1, 0]].permute(2, 0, 1).unsqueeze(0) / 255.0
        else:  # keep the old input so the upstream pretrained PPO still sees what it was trained on
            image_obs = image_obs.unsqueeze(0)
            image_obs = image_obs.permute(0,3,2,1)
        image_obs = self.conv_encoder(image_obs)
        navigation_obs = torch.tensor(observation[1], dtype=torch.float).to(self.device)
        observation = torch.cat((image_obs.view(-1), navigation_obs), -1)
        
        return observation