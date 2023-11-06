import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import transforms

import pytorch_lightning as pl
# from pytorch_lightning.metrics.functional import accuracy
from torchmetrics.functional import accuracy


class LitMNIST(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(28 * 28, 128)
        self.layer_2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.layer_1(x))
        x = self.layer_2(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        # acc = accuracy(y_hat, y,task='multiclass')
        
        
        self.log('train_loss_logginh', loss, on_step=True, on_epoch=True)
        # self.log('train_acc', acc, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

# prepare data
transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
mnist_train = MNIST('data', train=True, download=True, transform=transform)
mnist_train, mnist_val = random_split(mnist_train, [55000, 5000])
mnist_train = torch.utils.data.Subset(mnist_train, range(0, 100))

train_loader = DataLoader(mnist_train, batch_size=32)
val_loader = DataLoader(mnist_val, batch_size=32)

# train model
trainer = pl.Trainer(max_epochs=10,log_every_n_steps=1)
model = LitMNIST()
trainer.fit(model, train_loader)

