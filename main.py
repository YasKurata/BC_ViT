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

import wandb
import numpy as np
import argparse

import datetime
import pytz

from BC_model import BCmodel


# Parameters

parser = argparse.ArgumentParser()

parser.add_argument('--deg', type=float, default=0)
parser.add_argument('--tra', type=float, default=0)
parser.add_argument('--scl', type=float, default=0)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--do', type=float, default=0)
parser.add_argument('--wd', type=float, default=1e-3)
parser.add_argument('--posw', type=int, default=1)
parser.add_argument('--batch', type=int, default=16)
parser.add_argument('--hdn', type=int, default=500)
parser.add_argument('--update', type=int, default=0)
parser.add_argument('--index', type=str, default=0)
parser.add_argument('--cv-id', type=str, default=0)


args = parser.parse_args()

params = {"model_name": "vit_base_patch16_224",
          "optimizer": "Adam",
          "lr": args.lr,
          "batch_size": args.batch,
          "epochs": 1000,
          "img_size": 224,
          "weight_decay": args.wd,
          "num_of_update_params": args.update,
          "pos_weight": torch.tensor([args.posw]),
          "dropout": args.do,
          "num_workers": 1,
          "degree": args.deg,
          "translate": args.tra,
          "scale": args.scl,
          "hidden": args.hdn,
          "index": args.index,
          "CV-ID": args.cv_id,
          "fine_tuning_mode": False}


# Transform

img_size = params["img_size"]

train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((img_size, img_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomAffine(degrees=[params["degree"], params["degree"]], translate=(params["translate"], params["translate"]), scale=(1 - params["scale"], 1 + params["scale"]))
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((img_size, img_size))
])


# Datasets

def load_file(path):
    return np.load(path).astype(np.float32)

train_dataset = torchvision.datasets.DatasetFolder("data/Train_image_ex_3_sequence__val_only_center_no_test/CV{}/train/".format(args.index),
                                                   loader=load_file,
                                                   extensions="npy",
                                                   transform=train_transform)
val_dataset = torchvision.datasets.DatasetFolder("data/Train_image_ex_3_sequence__val_only_center_no_test/CV{}/val/".format(args.index),
                                                   loader=load_file,
                                                   extensions="npy",
                                                   transform=val_transform)

# Dataloader

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=params["batch_size"],
                                          num_workers=params["num_workers"], shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=params["batch_size"],
                                          num_workers=params["num_workers"], shuffle=False)


# Model

model_name=params["model_name"]


# Train

dt_now = datetime.datetime.now(pytz.timezone('Asia/Tokyo'))

logger = WandbLogger(project="{}_bcmodel".format(params["model_name"]), name=f"{dt_now.month}-{dt_now.day}-{dt_now.hour}-{dt_now.minute}-{dt_now.second}")
logger.log_hyperparams(params=params)

checkpoint_callback = ModelCheckpoint(dirpath="dir_path",
                                    filename="{epoch}-{v_loss:.3f}-{v_score:.3f}",
                                    monitor="v_loss",
                                    save_last=True,
                                    save_top_k=0,
                                    mode="min")

early_stopping =EarlyStopping(monitor="v_loss", patience=150, mode="min")


model = BCmodel(model_name=model_name,
    num_of_update_params=params["num_of_update_params"],
    lr=params["lr"],
    weight_decay=params["weight_decay"],
    pos_weight=params["pos_weight"])


trainer = pl.Trainer(gpus=1, logger=logger,
                    log_every_n_steps=10,
                    callbacks= [checkpoint_callback, early_stopping],
                    max_epochs=params["epochs"])

trainer.fit(model, train_loader, val_loader)
