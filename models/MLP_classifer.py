import torch
import torch.nn as nn
from torch.autograd import Variable


class classifier(nn.Module):
    def __init__(self, latent_dim, num_classes, type="A"):
        super().__init__()  # Call the init function of nn.Module
        if type == "A":
            self.classify = nn.Sequential(
                #nn.BatchNorm1d(latent_dim),
                nn.Linear(latent_dim, num_classes)
                # nn.Linear(latent_dim, latent_dim//2),
                # nn.ReLU(True),
                # nn.Dropout(p=0.5),
                # nn.Linear(latent_dim//2, latent_dim//2),
                # nn.ReLU(True),
                # nn.Dropout(p=0.5),
                # nn.Linear(latent_dim//2, num_classes),
            )

    def forward(self, x):
        x = self.classify(x)
        return x
        #return nn.functional.softmax(x, dim=1)
