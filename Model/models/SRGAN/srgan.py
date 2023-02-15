from IPython.display import clear_output
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt

import torch
from torch import nn
from torchvision.transforms import Resize
from torchvision.utils import make_grid

CHANNELS = 64
RES_BLOCK = 16
SUBSIZE = 0
NOISE_LEVEL = 0.15
SKIP_STRENTH = 1
ACTIVATION_STRENTH = 0.2
LOSS = nn.BCELoss()  # nn.L1Loss(), nn.MSELoss()
GEN_LEARN_PARAM = (0.0001, (0.5, 0.999))    #0.0001
DISC_LEARN_PARAM = (0.0001, (0.5, 0.999))  #0.00001

LOAD_MODEL = False
SAVE_MODEL = False

MODEL_FOLDER = Path(__file__).parent
GEN_PATH = MODEL_FOLDER / "generator.pth"
DIS_PATH = MODEL_FOLDER / "discriminator.pth"

CUDA = True
device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu") if CUDA else torch.device("cpu")

def conv_block(in_chan=CHANNELS, out_chan=CHANNELS, k_size=3, stride=1, norm=False, dropout=False, up=False):
    pad = "same" if stride == 1 else 0
    block = []
    if up:
        block.append(nn.Upsample(scale_factor=2))
    block.append(nn.Conv2d(in_channels=in_chan, out_channels=out_chan, kernel_size=k_size, stride=stride, padding=pad))
    if norm:
        block.append(nn.BatchNorm2d(out_chan, momentum=0.4))
    block.append(nn.LeakyReLU(ACTIVATION_STRENTH))
    if dropout:
        block.append(nn.Dropout(p=0.3))
    return nn.Sequential(*block)


class Generator(nn.Module):
    def __init__(self, n_res=RES_BLOCK, depth=CHANNELS, subsize=SUBSIZE) -> None:
        super().__init__()
        self.conv_1 = conv_block(3, depth, k_size=9, stride=1, norm=False, dropout=False, up=False)
        self.conv_2 = conv_block(depth, depth, k_size=3, stride=1, norm=False, dropout=False, up=False)

        self.resblock = nn.ModuleList()
        for _ in range(n_res):
            self.resblock.append(conv_block(depth, depth, k_size=3, stride=1, norm=True, dropout=False, up=False))
            self.resblock.append(conv_block(depth, depth, k_size=3, stride=1, norm=True, dropout=False, up=False))

        if subsize > 0:
            self.up_block = nn.ModuleList()
            self.up_block.append(conv_block(depth, 256, 3, 1, norm=False, dropout=False, up=True))
            for _ in range(subsize - 1):
                self.up_block.append(conv_block(256, 256, 3, 1, norm=False, dropout=False, up=True))
            self.up_block.append(conv_block(256, depth, 3, 1, norm=False, dropout=False, up=False))
        else:
            self.up_block = []

        self.last_conv = nn.Conv2d(in_channels=depth, out_channels=3, kernel_size=9, stride=1, padding='same')
        self.last_activ = nn.Sigmoid()

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        for i in range(0, len(self.resblock), 2):
            temp = x
            x = self.resblock[i](x)
            x = self.resblock[i+1](x)
            x = torch.add(x, temp, alpha=SKIP_STRENTH)
        for i in range(len(self.up_block)):
            x = self.up_block[i](x)
        x = self.last_conv(x)
        x = self.last_activ(x)
        return x
    

class Discriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.discrim = nn.Sequential(
            conv_block(3, 64, k_size=3, stride=2, norm=True, dropout=False, up=False),
            conv_block(64, 128, k_size=3, stride=2, norm=True, dropout=False, up=False),
            conv_block(128, 256, k_size=3, stride=2, norm=True, dropout=False, up=False),
            conv_block(256, 512, k_size=3, stride=2, norm=True, dropout=False, up=False),
            nn.Conv2d(512, 1, kernel_size=5, stride=1, padding="same"),
            nn.Flatten(),
            #nn.LazyLinear(out_features=256),
            #nn.LeakyReLU(0.2),
            #nn.Linear(in_features=256, out_features=1),
            nn.LazyLinear(out_features=1),
            nn.Sigmoid())

    def forward(self, x):
        return self.discrim(x)


def train(dataloader, load=LOAD_MODEL, save=SAVE_MODEL, sub=0, show_example=False):
    generator = Generator(subsize=sub).to(device)
    discriminator = Discriminator().to(device)

    loss = LOSS
    if CUDA:
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        loss = loss.cuda()

    g_optim = torch.optim.Adam(generator.parameters(), lr=GEN_LEARN_PARAM[0])#, betas=GEN_LEARN_PARAM[1])
    d_optim = torch.optim.Adam(discriminator.parameters(), lr=DISC_LEARN_PARAM[0])#, betas=DISC_LEARN_PARAM[1])
        
    if load:
        generator.load_state_dict(torch.load(GEN_PATH))
        discriminator.load_state_dict(torch.load(DIS_PATH))

    g_hist = []
    d_hist = []
    print_rate = (len(dataloader) // 10) if len(dataloader) > 10 else 1

    for i, data in enumerate(dataloader, 0):
        train_survey, train_obs = data
        train_survey_up = Resize(train_obs.shape[2:4])(train_survey) if sub > 0 else train_survey
        
        # Data is already load on GPU if pin memory False
        if dataloader.pin_memory:
            train_survey = train_survey.to(device)
            train_survey_up = train_survey_up.to(device)
            train_obs = train_obs.to(device)

        # Adding some level of noise to the discriminator output
        batch = train_obs.size(0)
        true_label = torch.full((batch,), 0, dtype=torch.float, device=device)
        false_label = torch.full((batch,), 1, dtype=torch.float, device=device)
        noise_label = NOISE_LEVEL * torch.rand(batch, dtype=torch.float, device=device)

        # Train discriminator
        discriminator.train()
        discriminator.zero_grad()
        label = torch.cat((true_label + noise_label, false_label - noise_label))
        d_output = torch.cat((discriminator(train_obs), discriminator(train_survey_up))).view(-1)
        d_loss = loss(d_output, label)
        d_loss.backward() # computes gradient, dloss/dx for every parameter x 
        # Update discriminator
        d_optim.step() 

        # Train generator
        discriminator.eval()  # Not train the discriminator in the generator phase
        generator.zero_grad()
        train_gen = generator(train_survey)
        gen_output = discriminator(train_gen).view(-1)
        g_loss = loss(gen_output, true_label)
        g_loss.backward()
        # Update generator
        g_optim.step()

        # Save history
        g_hist.append(g_loss.item())
        d_hist.append(d_loss.item())

        # Report every 10% of the work made
        if i % print_rate == 0:
            print(f"[{i} / {len(dataloader) - 1} | {round((i*100) / (len(dataloader) - 1))}%] D_loss: {d_loss.item():.3f} | G_loss: {g_loss.item():.3f}")
            if show_example:
                plt.imshow(np.transpose(make_grid([train_survey[0], train_gen[0], train_obs[0]], padding=2, nrow=3, normalize=True).cpu(),(1,2,0)))
                plt.show()
                clear_output(wait=True)

        if i % 1000 == 0 and save:
            torch.save(generator.state_dict(), GEN_PATH)
            torch.save(discriminator.state_dict(), DIS_PATH)

    # End step
    print(f"[{i} / {len(dataloader) - 1} | {round((i*100) / (len(dataloader) - 1))}%] D_loss: {d_loss.item():.3f} | G_loss: {g_loss.item():.3f}")
    if show_example:
        plt.imshow(np.transpose(make_grid([train_survey[0], train_gen[0], train_obs[0]], padding=2, nrow=3, normalize=True).cpu(),(1,2,0)))
        plt.show()
    if save:
        torch.save(generator.state_dict(), GEN_PATH)
        torch.save(discriminator.state_dict(), DIS_PATH)
    return (g_hist, d_hist)


def generate(img_list, device=device):
    generator = Generator().to(device)
    generator.load_state_dict(torch.load(GEN_PATH))
    with torch.no_grad():
        generated_img = []
        for img in img_list:
            generated_img.append(generator(img).detach())
    return generated_img
