import os
import torch
import datetime
from typing import Dict
import numpy as np
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm_notebook as tqdm
from matplotlib import pyplot as plt
import seaborn as sns
from sigma.src.utils import FeatureDataset

import warnings

warnings.filterwarnings("ignore")


class Experiment(object):
    def __init__(
        self,
        descriptor: str,
        general_results_dir: str,
        model: torch,
        model_args: dict,
        chosen_dataset: np,
        save_model_every_epoch=False,
    ):

        self.descriptor = "Model-" + descriptor
        print(f"model_name: {self.descriptor}")
        self.general_results_dir = general_results_dir
        self.set_up_results_dirs()  # Results dirs for output files and saved models
        self.chosen_dataset = chosen_dataset
        print(f"size_dataset: {self.chosen_dataset.shape}")
        self.save_model_every_epoch = save_model_every_epoch

        # Set Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"device: {str(self.device)}")

        # Set Model
        self.model_name = model.__name__
        self.model = model(in_channel=self.chosen_dataset[0].shape[-1], **model_args)
        print("num_parameters:", sum(p.numel() for p in self.model.parameters()))
        self.model.to(self.device)
        self.optimizer = Adam(self.model.parameters())
        self.noise_lambda = 0.00

    ### Methods ###
    def set_up_results_dirs(self):
        if not os.path.isdir(self.general_results_dir):
            os.mkdir(self.general_results_dir)

        self.date_and_descriptor = (
            datetime.datetime.today().strftime("%Y-%m-%d") + "_" + self.descriptor
        )
        self.results_dir = os.path.join(
            self.general_results_dir, self.date_and_descriptor
        )

        if not os.path.isdir(self.results_dir):
            os.mkdir(self.results_dir)

        self.params_dir = os.path.join(self.results_dir, "params")
        if not os.path.isdir(self.params_dir):
            os.mkdir(self.params_dir)

        # self.backup_dir = os.path.join(self.results_dir,'backup')
        # if not os.path.isdir(self.backup_dir):
        #     os.mkdir(self.backup_dir)

    def run_model(
        self,
        num_epochs: int,
        batch_size: int,
        patience: int = 50,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.0,
        task: str = "train_eval",
        noise_added=None,
        criterion: str = "MSE",
        KLD_lambda:float = 1e-4,
        print_latent: bool = False,
        lr_scheduler_args: Dict={
                            "factor": 0.5,
                            "verbose": True,
                            "patience": 5,
                            "threshold": 1e-2,
                            "min_lr": 1e-7,
                        },
    ):
        # Loss functions
        if criterion == "MSE":
            self.criterion = nn.MSELoss()
        elif criterion == "BCE":
            self.criterion = nn.BCELoss()
        elif criterion == "L1":
            self.criterion = nn.L1Loss()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.KLD_lambda = KLD_lambda
        self.num_epochs = num_epochs
        print(f"num_epochs: {self.num_epochs}")
        self.batch_size = batch_size
        print(f"batch_size: {self.batch_size}")

        self.noise = noise_added
        self.print_latent = print_latent

        # Set Task
        self.task = task
        assert self.task in ["train_eval", "train_all"]
        print(f"task: {self.task}")

        # Data
        if self.task == "train_eval":
            self.dataset_train = FeatureDataset(
                self.chosen_dataset, "train", self.noise
            )
            self.dataset_test = FeatureDataset(self.chosen_dataset, "test")
        elif self.task == "train_all":
            self.dataset_train = FeatureDataset(self.chosen_dataset, "all", self.noise)
            # self.dataset_test = utils.FeatureDataset(self.chosen_dataset,'test')

        # Tracking losses and evaluation results
        self.train_loss = np.zeros((self.num_epochs + 1))
        self.test_loss = np.zeros((self.num_epochs + 1))

        # For early stopping
        self.initial_patience = patience
        self.patience_remaining = patience
        self.best_valid_epoch = 0
        self.min_test_loss = np.inf

        # Set Optimozer
        self.optimizer = Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        self.scheduler = ReduceLROnPlateau(self.optimizer, **lr_scheduler_args)

        print(
            f"optimizer: lr={str(self.learning_rate)}",
            f"and weight_decay={str(self.weight_decay)}\n",
        )

        print("Start training ...\n")

        start_epoch = 0
        if self.task == "train_eval":
            train_dataloader = DataLoader(
                self.dataset_train, batch_size=self.batch_size, shuffle=True
            )
            test_dataloader = DataLoader(
                self.dataset_test, batch_size=4096, shuffle=False
            )
        elif self.task == "train_all":
            train_dataloader = DataLoader(
                self.dataset_train, batch_size=self.batch_size, shuffle=True
            )

        for epoch in range(
            start_epoch, self.num_epochs
        ):  # loop over the dataset multiple times

            self.train(train_dataloader, epoch + 1)

            if self.task == "train_eval":
                self.test(test_dataloader, epoch + 1)
            elif self.task == "train_all":
                self.scheduler.step(self.train_loss[epoch + 1])
                self.early_stopping_check(epoch + 1)

            if self.save_model_every_epoch:
                self.save_model(epoch + 1)

            if self.patience_remaining <= 0:
                print(
                    "No more patience (",
                    self.initial_patience,
                    ") left at epoch",
                    epoch + 1,
                )
                print(
                    "--> Implementing early stopping. Best epoch was:",
                    self.best_valid_epoch,
                )
                break

        # self.save_final_summary()

    def train(self, dataloader, epoch):
        self.model.train()
        if self.model_name == "AutoEncoder":
            epoch_loss = self.iterate_through_batches(
                self.model, dataloader, epoch, training=True
            )
        elif self.model_name == "VariationalAutoEncoder":
            epoch_loss = self.iterate_through_batches_VAE(
                self.model, dataloader, epoch, training=True
            )

        self.train_loss[epoch] = epoch_loss

    def test(self, dataloader, epoch):
        self.model.eval()
        with torch.no_grad():
            if self.model_name == "AutoEncoder":
                epoch_loss = self.iterate_through_batches(
                    self.model, dataloader, epoch, training=False
                )
            elif self.model_name == "VariationalAutoEncoder":
                epoch_loss = self.iterate_through_batches_VAE(
                    self.model, dataloader, epoch, training=False
                )
        self.test_loss[epoch] = epoch_loss
        self.scheduler.step(epoch_loss)
        self.early_stopping_check(epoch)

    def early_stopping_check(self, epoch):
        """Check whether criteria for early stopping are met and update
        counters accordingly"""
        if self.task == "train_eval":
            test_loss = self.test_loss[epoch]
        elif self.task == "train_all":
            test_loss = self.train_loss[epoch]

        if (
            (test_loss < self.min_test_loss)
            or epoch == 0
            or epoch == self.num_epochs - 1
        ):  # then save parameters
            self.min_test_loss = test_loss
            if not self.save_model_every_epoch:
                self.save_model(epoch)
            self.best_valid_epoch = epoch
            self.patience_remaining = self.initial_patience
            print(
                f"Epoch {epoch} ----> model saved, train_loss={self.train_loss[epoch]:.6f} | test_loss = {test_loss:.6f}"
            )
        else:
            self.patience_remaining -= 1

    def save_model(self, epoch):
        if self.print_latent:
            self.plot_latent(epoch)

        check_point = {
            "params": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "train_loss": self.train_loss,
            "test_loss": self.test_loss,
        }
        torch.save(
            check_point,
            os.path.join(self.params_dir, self.descriptor + f"_epoch{epoch:03}"),
        )

    def plot_latent(self, epoch):
        latent = self.get_latent()
        fig, axs = plt.subplots(1, 1, figsize=(3, 3), dpi=100)
        sns.scatterplot(latent[:, 0], latent[:, 1], s=0.5, alpha=0.1, ax=axs, color="r")
        # axs.set_aspect(1)
        axs.set_title(f"Epoch{epoch+1}")
        plt.show()

    def KLD_loss(self, mu, logvar):
        loss = -0.5 * torch.mean(1 + logvar - mu ** 2 - logvar.exp())
        return loss

    def iterate_through_batches(self, model, dataloader, epoch, training):
        epoch_loss = list()

        # Initialize numpy arrays for storing results. examples x labels
        # Do NOT use concatenation, or else you will have memory fragmentation.
        disable = False if training else True
        with tqdm(dataloader, unit="batch", disable=disable, leave=False) as tepoch:
            for batch_idx, batch in enumerate(tepoch):
                if training:
                    tepoch.set_description(
                        f"Epoch [ {epoch:02} / {self.num_epochs:02} ]"
                    )
                x = batch.to(self.device)

                self.optimizer.zero_grad()
                if training:
                    recon_x = model(x)
                else:
                    with torch.set_grad_enabled(False):
                        recon_x = model(x)

                loss = self.criterion(recon_x, x)

                if training:
                    loss.backward()
                    self.optimizer.step()

                epoch_loss.append(loss.detach().item())
                torch.cuda.empty_cache()

            avg_loss = sum(epoch_loss) / len(epoch_loss)
            if training:
                tepoch.set_postfix(train_loss=f"{avg_loss:.6f}")

        # Return loss and classification predictions and classification gr truth
        return sum(epoch_loss) / len(epoch_loss)

    def iterate_through_batches_VAE(self, model, dataloader, epoch, training):
        epoch_loss = list()

        # Initialize numpy arrays for storing results. examples x labels
        # Do NOT use concatenation, or else you will have memory fragmentation.
        disable = False if training else True
        with tqdm(dataloader, unit="batch", disable=disable, leave=False) as tepoch:
            for batch_idx, batch in enumerate(tepoch):
                if training:
                    tepoch.set_description(
                        f"Epoch [ {epoch:02} / {self.num_epochs:02} ]"
                    )
                x = batch.to(self.device)

                self.optimizer.zero_grad()
                if training:
                    mu, logvar, z, x_recon = model(x)
                else:
                    with torch.set_grad_enabled(False):
                        mu, logvar, z, x_recon = model(x)

                recon_loss = self.criterion(x_recon, x)
                KLD_loss = self.KLD_loss(mu, logvar)
                loss = (
                    self.criterion(x_recon, x)
                    + self.KLD_loss(mu, logvar) * self.KLD_lambda
                )

                if training:
                    loss.backward()
                    self.optimizer.step()

                epoch_loss.append(loss.detach().item())
                torch.cuda.empty_cache()

            avg_loss = sum(epoch_loss) / len(epoch_loss)
            if training:
                tepoch.set_postfix(train_loss=f"{avg_loss:.6f}")

        # Return loss and classification predictions and classification gr truth
        return sum(epoch_loss) / len(epoch_loss)

    def load_trained_model(self, old_model_path):
        print(f"Loading model parameters from {old_model_path}")
        self.old_model_path = old_model_path
        check_point = torch.load(self.old_model_path)
        self.model.load_state_dict(check_point["params"])
        self.optimizer.load_state_dict(check_point["optimizer"])
        if "train_loss" in check_point.keys():
            self.train_loss = check_point["train_loss"]
            self.test_loss = check_point["test_loss"]

    def get_latent(self) -> np:
        latents = list()
        dataset_ = FeatureDataset(self.chosen_dataset, "all")
        loader = DataLoader(dataset_, batch_size=4096, shuffle=False)

        with torch.no_grad():
            self.model.eval()
            for i, data in enumerate(loader):
                x = data.to(self.device)
                z = self.model._encode(x)
                latents.append(z.detach().cpu().numpy())

        return np.concatenate(latents, axis=0)
