import torch, torchvision, torchmetrics
import torch.nn as nn
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger

from torchinfo import summary
import timm

from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from collections import OrderedDict

import numpy as np


class BCmodel(pl.LightningModule):
    def __init__(self, model_name, num_of_update_params, lr, weight_decay, pos_weight):
        super(BCmodel, self).__init__()  
        
        self.num_of_update_params = num_of_update_params
        self.model_name = model_name
        self.lr = lr
        self.batch_size = params["batch_size"]
        self.wd = weight_decay
        self.pos_weight = pos_weight
        
        
        self.model = timm.create_model(model_name=self.model_name, pretrained=True)
        n_features = self.model.num_features
        
        self.model.head = nn.Sequential(nn.Linear(n_features, args.hdn),
        nn.ReLU(), nn.Dropout(p=args.do),
        nn.Linear(args.hdn, 100),
        nn.ReLU(), nn.Dropout(p=args.do),
        nn.Linear(100, 1))
        
        if params["fine_tuning_mode"] == False:
            param_names_list = []
            for name, _ in self.model.named_parameters():
                param_names_list.append(name)
            
            update_param_names = param_names_list[- num_of_update_params:]
            
            for name, param in self.model.named_parameters():
                if name in update_param_names:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        
        self.save_hyperparameters()
        
        self.train_acc = torchmetrics.Accuracy(task='binary')
        self.val_acc = torchmetrics.Accuracy(task='binary')

    def forward(self, data):
        pred = self.model(data)
        return pred

    def training_step(self, batch, batch_idx):
        img, label = batch
        label = label.float()

        logits = self(img).squeeze(1)
        loss = self.loss_fn(logits, label)       
        preds = logits.sigmoid()     
        acc = self.train_acc(preds, label.int())      
        output = OrderedDict({
            "targets": label.detach(), "preds": preds.detach(), "loss": loss, "train_acc_step": acc
        })
        
        return output
    
    def training_epoch_end(self, outputs):
        d = dict()
        d["epoch"] = int(self.current_epoch)
        d["t_loss"] = torch.stack([o["loss"] for o in outputs]).mean().item()

        targets = torch.cat([o["targets"].view(-1) for o in outputs]).cpu().numpy()
        preds = torch.cat([o["preds"].view(-1) for o in outputs]).cpu().numpy()

        score = roc_auc_score(y_true=targets, y_score=preds)
        t_acc = self.train_acc.compute()
        
        d["t_score"] = score      
        d["t_acc"] = t_acc
        
        self.log_dict(d, prog_bar=True)
        
    def validation_step(self, batch, batch_idx):
        img, label = batch
        label = label.float()
        logits = self(img).squeeze(1)
        loss = self.loss_fn(logits, label)
        preds = logits.sigmoid()
        
        output = OrderedDict({
            "targets": label.detach(), "preds": preds.detach(), "loss": loss.detach()
        })      
        acc = self.val_acc(preds, label.int())
        
        return output
    
    def validation_epoch_end(self, outputs):
        d = dict()
        d["epoch"] = int(self.current_epoch)
        d["v_loss"] = torch.stack([o["loss"] for o in outputs]).mean().item()

        targets = torch.cat([o["targets"].view(-1) for o in outputs]).cpu().numpy()
        preds = torch.cat([o["preds"].view(-1) for o in outputs]).cpu().numpy()

        target_list = list(targets)
        preds_list = list(preds)
        final_preds_list = [1 if x>0.5 else 0 for x in preds_list]

        score = roc_auc_score(y_true=targets, y_score=preds)
        acc = accuracy_score(target_list, final_preds_list)

        d["v_score"] = score
        d["v_acc"] = acc
        
        self.log_dict(d, prog_bar=True)

    def configure_optimizers(self):
        optimizer = self.optimizer
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=30, factor=0.5, verbose=True)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "v_loss"}