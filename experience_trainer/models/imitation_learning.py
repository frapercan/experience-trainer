import pytorch_lightning as pl
import torch
import torch.optim as optim
from torchvision.models import densenet201, DenseNet201_Weights
from torchmetrics import Accuracy
from experience_trainer.focal import FocalLoss




class DensenetIL(pl.LightningModule):
    def __init__(self,config, metadata, criterion=FocalLoss(), learning_rate=0.001):
        super(DensenetIL, self).__init__()
        self.config = config

        self.criterion = criterion
        self.learning_rate = learning_rate
        self.actions = metadata['actions']

        self.example_input_array = torch.randn((1, 3, 224, 224))

        self.accuracy = Accuracy(task="multiclass", num_classes=len(self.actions), top_k=1)
        self.accuracy_val = Accuracy(task="multiclass", num_classes=len(self.actions), top_k=1)

        self.model = densenet201(weights=DenseNet201_Weights.DEFAULT)
        # Modify the last layer for num_actions
        num_ftrs = self.model.classifier.in_features
        self.model.classifier = torch.nn.Linear(num_ftrs, len(self.actions))

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        state, action = batch
        output = self.forward(state)
        loss = self.criterion(output, action)
        acc = self.accuracy(output.argmax(dim=1), action.argmax(dim=1))

        self.log('loss', loss, prog_bar=True)
        self.log('train_acc_step', acc, prog_bar=True)

        return loss

    def on_train_epoch_end(self):
        self.log('train_acc_epoch', self.accuracy, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        state, action = batch
        output = self.forward(state)
        loss = self.criterion(output, action)
        val_acc = self.accuracy_val(output.argmax(dim=1), action.argmax(dim=1))


        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc_step', val_acc, prog_bar=True)

        return loss

    def on_validation_epoch_end(self):
        self.log('val_acc_epoch', self.accuracy_val, prog_bar=True)



    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        if self.config['CosineAnnealingLR']:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, self.config['CosineAnnealingLR_steps'])
            return [optimizer], [scheduler]

        else:
            return optimizer


def imshow(tensor):
    # Convertir tensor de CHW a HWC
    img = tensor.cpu().permute(1, 2, 0).numpy()
    plt.imshow(img)
    plt.show()
