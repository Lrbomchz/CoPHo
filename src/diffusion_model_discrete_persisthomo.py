import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import time
import wandb
import os

from models.transformer_model import GraphTransformer
from diffusion.noise_schedule import DiscreteUniformTransition, PredefinedNoiseScheduleDiscrete, \
    MarginalUniformTransition
from src.diffusion import diffusion_utils
from metrics.train_metrics import TrainLossDiscrete
from metrics.abstract_metrics import SumExceptBatchMetric, SumExceptBatchKL, NLL
from src import utils
from src import condition_utils

import numpy as np
from models.GNN_model import GraphDistanceModel
from models import condi_config

#test
from data import GraphDataset, collate_fn
from torch.utils.data import DataLoader
import pickle

from hydra.utils import to_absolute_path, get_original_cwd

class DiscreteDenoisingDiffusion(pl.LightningModule):
    def __init__(self, cfg, dataset_infos, train_metrics, sampling_metrics, visualization_tools, extra_features,
                 domain_features, guidance_model=None):
        super().__init__()
        # self.automatic_optimization = False

        input_dims = dataset_infos.input_dims
        output_dims = dataset_infos.output_dims
        nodes_dist = dataset_infos.nodes_dist

        self.cfg = cfg
        self.name = cfg.general.name
        self.model_dtype = torch.float32
        self.T = cfg.model.diffusion_steps

        self.Xdim = input_dims['X']
        self.Edim = input_dims['E']
        self.ydim = input_dims['y']
        self.Xdim_output = output_dims['X']
        self.Edim_output = output_dims['E']
        self.ydim_output = output_dims['y']

        print(input_dims['y'])
        print(output_dims['y'])

        self.node_dist = nodes_dist
        self.i = 0

        self.dataset_info = dataset_infos

        self.train_loss = TrainLossDiscrete(self.cfg.model.lambda_train)

        self.val_nll = NLL()
        self.val_X_kl = SumExceptBatchKL()
        self.val_E_kl = SumExceptBatchKL()
        self.val_X_logp = SumExceptBatchMetric()
        self.val_E_logp = SumExceptBatchMetric()

        self.test_nll = NLL()
        self.test_X_kl = SumExceptBatchKL()
        self.test_E_kl = SumExceptBatchKL()
        self.test_X_logp = SumExceptBatchMetric()
        self.test_E_logp = SumExceptBatchMetric()

        self.train_metrics = train_metrics
        self.sampling_metrics = sampling_metrics

        self.visualization_tools = visualization_tools
        self.extra_features = extra_features
        self.domain_features = domain_features

        self.model = GraphTransformer(n_layers=cfg.model.n_layers,
                                      input_dims=input_dims,
                                      hidden_mlp_dims=cfg.model.hidden_mlp_dims,
                                      hidden_dims=cfg.model.hidden_dims,
                                      output_dims=output_dims,
                                      act_fn_in=nn.ReLU(),
                                      act_fn_out=nn.ReLU())

        self.noise_schedule = PredefinedNoiseScheduleDiscrete(cfg.model.diffusion_noise_schedule,
                                                              timesteps=cfg.model.diffusion_steps)

        if cfg.model.transition == 'uniform':
            self.transition_model = DiscreteUniformTransition(x_classes=self.Xdim_output, e_classes=self.Edim_output,
                                                              y_classes=self.ydim_output)
            x_limit = torch.ones(self.Xdim_output) / self.Xdim_output
            e_limit = torch.ones(self.Edim_output) / self.Edim_output
            y_limit = torch.ones(self.ydim_output) / self.ydim_output
            self.limit_dist = utils.PlaceHolder(X=x_limit, E=e_limit, y=y_limit)
        elif cfg.model.transition == 'marginal':

            node_types = self.dataset_info.node_types.float()
            x_marginals = node_types / torch.sum(node_types)

            edge_types = self.dataset_info.edge_types.float()
            e_marginals = edge_types / torch.sum(edge_types)
            print(f"Marginal distribution of the classes: {x_marginals} for nodes, {e_marginals} for edges")
            self.transition_model = MarginalUniformTransition(x_marginals=x_marginals, e_marginals=e_marginals,
                                                              y_classes=self.ydim_output)
            self.limit_dist = utils.PlaceHolder(X=x_marginals, E=e_marginals,
                                                y=torch.ones(self.ydim_output) / self.ydim_output)

        self.save_hyperparameters(ignore=['train_metrics', 'sampling_metrics'])
        self.start_epoch_time = None
        self.train_iterations = None
        self.val_iterations = None
        self.log_every_steps = cfg.general.log_every_steps
        self.number_chain_steps = cfg.general.number_chain_steps
        self.best_val_nll = 1e8
        self.val_counter = 0

        self.guidance_model = guidance_model
        self.enable_condition = condi_config.enable_condition
        self.condi_delta = condi_config.condi_delta
        self.sampling_multiplier = condi_config.sampling_multiplier
        self.condition_phase_s = condi_config.condition_phase_s
        self.condition_phase_e = condi_config.condition_phase_e
        self.all_condition_phase_s = condi_config.all_condition_phase_s
        self.all_condition_phase_e = condi_config.all_condition_phase_e
        self.homo_condition_phase_s = condi_config.homo_condition_phase_s
        self.homo_condition_phase_e = condi_config.homo_condition_phase_e
        self.condition_mode = condi_config.condition_mode
        self.condition_weight = condi_config.condition_weight
        self.condition_target = condi_config.condition_target
        self.flag = 0
        if self.enable_condition:
            self.name = (f"{self.name}|delta{self.condi_delta}|weight{self.condition_weight}|"
                                f"multiplier{self.sampling_multiplier}|phase{self.condition_phase_s}-{self.condition_phase_e}|"
                                f"mode{self.condition_mode}")


    def assign_guidance_model(self, gmodel):
        gmodel.eval()
        gmodel.to(self.device)
        self.guidance_model = gmodel
        print("Successfully assigned guidance model")


    def training_step(self, data, i):
        if data.edge_index.numel() == 0:
            self.print("Found a batch with no edges. Skipping.")
            return
        # 正常的前向传播
        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        dense_data = dense_data.mask(node_mask)
        X, E = dense_data.X, dense_data.E
        noisy_data = self.apply_noise(X, E, data.y, node_mask)
        extra_data = self.compute_extra_data(noisy_data)
        pred = self.forward(noisy_data, extra_data, node_mask)
        loss = self.train_loss(masked_pred_X=pred.X, masked_pred_E=pred.E, pred_y=pred.y,
                               true_X=X, true_E=E, true_y=data.y,
                               log=i % self.log_every_steps == 0)

        self.train_metrics(masked_pred_X=pred.X, masked_pred_E=pred.E, true_X=X, true_E=E,
                           log=i % self.log_every_steps == 0)

        print(f"Loss: {loss.item()}")

        return {'loss': loss}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.train.lr, amsgrad=True,
                                      weight_decay=self.cfg.train.weight_decay)
        return optimizer

    def on_fit_start(self) -> None:
        self.train_iterations = len(self.trainer.datamodule.train_dataloader())
        self.print("Size of the input features", self.Xdim, self.Edim, self.ydim)
        if self.local_rank == 0:
            utils.setup_wandb(self.cfg)

    def on_train_epoch_start(self) -> None:
        self.print("Starting train epoch...")
        self.start_epoch_time = time.time()
        self.train_loss.reset()
        self.train_metrics.reset()

    def on_train_epoch_end(self) -> None:
        to_log = self.train_loss.log_epoch_metrics()
        self.print(f"Epoch {self.current_epoch}: X_CE: {to_log['train_epoch/x_CE'] :.3f}"
                   f" -- E_CE: {to_log['train_epoch/E_CE'] :.3f} --"
                   f" y_CE: {to_log['train_epoch/y_CE'] :.3f}"
                   f" -- {time.time() - self.start_epoch_time:.1f}s ")
        epoch_at_metrics, epoch_bond_metrics = self.train_metrics.log_epoch_metrics()
        self.print(f"Epoch {self.current_epoch}: {epoch_at_metrics} -- {epoch_bond_metrics}")
        if torch.cuda.is_available():
            print(torch.cuda.memory_summary())
        else:
            print("CUDA is not available. Skipping memory summary.")

    def on_validation_epoch_start(self) -> None:
        self.val_nll.reset()
        self.val_X_kl.reset()
        self.val_E_kl.reset()
        self.val_X_logp.reset()
        self.val_E_logp.reset()
        self.sampling_metrics.reset()

    def validation_step(self, data, i):
        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        dense_data = dense_data.mask(node_mask)
        noisy_data = self.apply_noise(dense_data.X, dense_data.E, data.y, node_mask)
        extra_data = self.compute_extra_data(noisy_data)
        pred = self.forward(noisy_data, extra_data, node_mask)
        nll = self.compute_val_loss(pred, noisy_data, dense_data.X, dense_data.E, data.y, node_mask, test=False)
        return {'loss': nll}

    def on_validation_epoch_end(self) -> None:
        metrics = [self.val_nll.compute(), self.val_X_kl.compute() * self.T, self.val_E_kl.compute() * self.T,
                   self.val_X_logp.compute(), self.val_E_logp.compute()]
        if wandb.run:
            wandb.log({"val/epoch_NLL": metrics[0],
                       "val/X_kl": metrics[1],
                       "val/E_kl": metrics[2],
                       "val/X_logp": metrics[3],
                       "val/E_logp": metrics[4]}, commit=False)

        self.print(f"Epoch {self.current_epoch}: Val NLL {metrics[0] :.2f} -- Val Atom type KL {metrics[1] :.2f} -- ",
                   f"Val Edge type KL: {metrics[2] :.2f}")

        # Log val nll with default Lightning logger, so it can be monitored by checkpoint callback
        val_nll = metrics[0]
        self.log("val/epoch_NLL", val_nll, sync_dist=True)

        if val_nll < self.best_val_nll:
            self.best_val_nll = val_nll
        self.print('Val loss: %.4f \t Best val loss:  %.4f\n' % (val_nll, self.best_val_nll))

        self.val_counter += 1
        if self.val_counter % self.cfg.general.sample_every_val == 0:
            start = time.time()
            samples_left_to_generate = self.cfg.general.samples_to_generate
            samples_left_to_save = self.cfg.general.samples_to_save
            chains_left_to_save = self.cfg.general.chains_to_save

            samples = []

            val_loader = self.trainer.datamodule.val_dataloader()
            for data in val_loader:
                y_cond = data.y.to(self.device)
                print("[DEBUG] val", y_cond.shape)
                break

            ident = 0
            while samples_left_to_generate > 0:
                bs = self.cfg.train.batch_size
                to_generate = min(samples_left_to_generate, bs, len(y_cond))
                to_save = min(samples_left_to_save, bs, len(y_cond))
                chains_save = min(chains_left_to_save, bs, len(y_cond))
                y_cond = y_cond[:to_generate]
                samples.extend(self.sample_batch(batch_id=ident, batch_size=to_generate, num_nodes=None,
                                                 save_final=to_save,
                                                 keep_chain=chains_save,
                                                 number_chain_steps=self.number_chain_steps, y_cond=y_cond))
                ident += to_generate

                samples_left_to_save -= to_save
                samples_left_to_generate -= to_generate
                chains_left_to_save -= chains_save
            self.print("Computing sampling metrics...")
            self.sampling_metrics.forward(samples, self.name, self.current_epoch, val_counter=-1, test=False,
                                          local_rank=self.local_rank)
            self.print(f'Done. Sampling took {time.time() - start:.2f} seconds\n')
            print("Validation epoch end ends...")

    def on_test_epoch_start(self) -> None:
        self.print("Starting test...")
        #self.test_nll.reset()
        #self.test_X_kl.reset()
        #self.test_E_kl.reset()
        #self.test_X_logp.reset()
        #self.test_E_logp.reset()
        if self.local_rank == 0:
            utils.setup_wandb(self.cfg)

    @torch.enable_grad()
    @torch.inference_mode(False)
    def test_step(self, data, i):
        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        dense_data = dense_data.mask(node_mask)
        noisy_data = self.apply_noise(dense_data.X, dense_data.E, data.y, node_mask)
        extra_data = self.compute_extra_data(noisy_data)
        #pred = self.forward(noisy_data, extra_data, node_mask)
        #nll = self.compute_val_loss(pred, noisy_data, dense_data.X, dense_data.E, data.y, node_mask, test=True)
        #return {'loss': nll}

    @torch.enable_grad()
    @torch.inference_mode(False)
    def on_test_epoch_end(self) -> None:
        """ Measure likelihood on a test set and compute stability metrics. """
        #metrics = [self.test_nll.compute(), self.test_X_kl.compute(), self.test_E_kl.compute(),
        #           self.test_X_logp.compute(), self.test_E_logp.compute()]
        #if wandb.run:
        #    wandb.log({"test/epoch_NLL": metrics[0],
        #               "test/X_kl": metrics[1],
        #               "test/E_kl": metrics[2],
        ##               "test/X_logp": metrics[3],
         #              "test/E_logp": metrics[4]}, commit=False)

        #self.print(f"Epoch {self.current_epoch}: Test NLL {metrics[0] :.2f} -- Test Atom type KL {metrics[1] :.2f} -- ",
        #           f"Test Edge type KL: {metrics[2] :.2f}")

        #test_nll = metrics[0]
        #if wandb.run:
         #   wandb.log({"test/epoch_NLL": test_nll}, commit=False)

        #self.print(f'Test loss: {test_nll :.4f}')

        #samples_left_to_generate = self.cfg.general.final_model_samples_to_generate
        #samples_left_to_save = self.cfg.general.final_model_samples_to_save
        samples_left_to_generate = 17
        samples_left_to_save = 17
        chains_left_to_save = self.cfg.general.final_model_chains_to_save

        print("samples_left_to_generate:", samples_left_to_generate)

        test_loader = self.trainer.datamodule.test_dataloader()
        for data in test_loader:
            cond = data.cond.to(self.device)
            #print(data.X.shape, data.edge_attr.shape)
            print("[DEBUG] cond shape", cond.shape)
            if cond.dim() == 1:
                cond = cond.unsqueeze(dim=1)
            break

        samples = []
        id = 0
        while samples_left_to_generate > 0:
            self.print(f'Samples left to generate: {samples_left_to_generate}/'
                       f'{self.cfg.general.final_model_samples_to_generate}', end='', flush=True)

            bs = self.cfg.train.batch_size
            to_generate = min(samples_left_to_generate, bs, len(cond))
            to_save = min(samples_left_to_save, bs, len(cond))
            chains_save = min(chains_left_to_save, bs, len(cond))
            cond = cond[:to_generate]

            all_chain_X = []  # 用于保存所有 batch 的 chain_X
            all_chain_E = []  # 用于保存所有 batch 的 chain_E

            molecule_list, chain_X, chain_E = self.sample_batch(id, to_generate, num_nodes=None, save_final=to_save,
                                             keep_chain=chains_save, number_chain_steps=self.number_chain_steps,
                                             y_cond=None, cond=cond)
            samples.extend(molecule_list)
            all_chain_X.append(chain_X)
            all_chain_E.append(chain_E)
            id += to_generate
            samples_left_to_save -= to_save
            samples_left_to_generate -= to_generate
            chains_left_to_save -= chains_save
        self.print("Saving the generated graphs")
        for i in range(1, 50):
            filename = (f"outputs/generated_samples|conditions-{self.condition_target[0]}|"
                        f"phase{self.condition_phase_s}-{self.condition_phase_e}|"
                        f"mode{self.condition_mode}-{i}.txt")
            filename =  to_absolute_path(filename)
            if os.path.exists(filename):
                continue
            else:
                break
        with open(filename, 'w') as f:
            for item in samples:
                f.write(f"N={item[0].shape[0]}\n")
                atoms = item[0].tolist()
                f.write("X: \n")
                for at in atoms:
                    f.write(f"{at} ")
                f.write("\n")
                f.write("E: \n")
                for bond_list in item[1]:
                    for bond in bond_list:
                        f.write(f"{bond} ")
                    f.write("\n")
                f.write("\n")
        self.print("Generated graphs Saved. Computing sampling metrics...")

        # 设置一个新的文件名，用于保存中间链信息
        for i in range(1, 50):
            chain_filename = (f"outputs/intermediate_chains_comm|conditions-{self.condition_target[0]}"
                              f"|phase{self.condition_phase_s}-{self.condition_phase_e}|"
                              f"mode{self.condition_mode}-{i}.txt")
            chain_filename =  to_absolute_path(chain_filename)

            if os.path.exists(chain_filename):
                continue
            else:
                break

        chain_counts = 0
        with open(chain_filename, 'w') as f:
            # 遍历所有 batch 的链信息，每个 chain_X 的 shape 大致为 (number_chain_steps+10, keep_chain, n_nodes)
            for batch_idx, (chain_X, chain_E) in enumerate(zip(all_chain_X, all_chain_E)):
                f.write(f"Batch {batch_idx}:\n")
                num_steps = chain_X.shape[0]
                for step in range(num_steps):
                    f.write(f"Step {step}:\n")
                    f.write("X:\n")
                    # 假设可以直接打印 tensor 的数据或者转换为 list
                    f.write(str(chain_X[step].tolist()) + "\n")
                    f.write("E:\n")
                    f.write(str(chain_E[step].tolist()) + "\n")
                    f.write("\n")
                    chain_counts += 1

        self.sampling_metrics(samples, self.name, self.current_epoch, self.val_counter, test=True,
                               local_rank=self.local_rank)
        # self.print(f"Write {chain_counts} noise graphs to file.")
        self.print("Done testing.")

    def kl_prior(self, X, E, node_mask):
        """Computes the KL between q(z1 | x) and the prior p(z1) = Normal(0, 1).

        This is essentially a lot of work for something that is in practice negligible in the loss. However, you
        compute it so that you see it when you've made a mistake in your noise schedule.
        """
        # Compute the last alpha value, alpha_T.
        ones = torch.ones((X.size(0), 1), device=X.device)
        Ts = self.T * ones
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_int=Ts)  # (bs, 1)

        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.device)

        # Compute transition probabilities
        probX = X @ Qtb.X  # (bs, n, dx_out)
        probE = E @ Qtb.E.unsqueeze(1)  # (bs, n, n, de_out)
        assert probX.shape == X.shape

        bs, n, _ = probX.shape

        limit_X = self.limit_dist.X[None, None, :].expand(bs, n, -1).type_as(probX)
        limit_E = self.limit_dist.E[None, None, None, :].expand(bs, n, n, -1).type_as(probE)

        # Make sure that masked rows do not contribute to the loss
        limit_dist_X, limit_dist_E, probX, probE = diffusion_utils.mask_distributions(true_X=limit_X.clone(),
                                                                                      true_E=limit_E.clone(),
                                                                                      pred_X=probX,
                                                                                      pred_E=probE,
                                                                                      node_mask=node_mask)

        kl_distance_X = F.kl_div(input=probX.log(), target=limit_dist_X, reduction='none')
        kl_distance_E = F.kl_div(input=probE.log(), target=limit_dist_E, reduction='none')

        return diffusion_utils.sum_except_batch(kl_distance_X) + \
            diffusion_utils.sum_except_batch(kl_distance_E)

    def compute_Lt(self, X, E, y, pred, noisy_data, node_mask, test):
        pred_probs_X = F.softmax(pred.X, dim=-1)
        pred_probs_E = F.softmax(pred.E, dim=-1)
        pred_probs_y = F.softmax(pred.y, dim=-1)

        Qtb = self.transition_model.get_Qt_bar(noisy_data['alpha_t_bar'], self.device)
        Qsb = self.transition_model.get_Qt_bar(noisy_data['alpha_s_bar'], self.device)
        Qt = self.transition_model.get_Qt(noisy_data['beta_t'], self.device)

        # Compute distributions to compare with KL
        bs, n, d = X.shape
        prob_true = diffusion_utils.posterior_distributions(X=X, E=E, y=y, X_t=noisy_data['X_t'], E_t=noisy_data['E_t'],
                                                            y_t=noisy_data['y_t'], Qt=Qt, Qsb=Qsb, Qtb=Qtb)
        prob_true.E = prob_true.E.reshape((bs, n, n, -1))
        prob_pred = diffusion_utils.posterior_distributions(X=pred_probs_X, E=pred_probs_E, y=pred_probs_y,
                                                            X_t=noisy_data['X_t'], E_t=noisy_data['E_t'],
                                                            y_t=noisy_data['y_t'], Qt=Qt, Qsb=Qsb, Qtb=Qtb)
        prob_pred.E = prob_pred.E.reshape((bs, n, n, -1))

        # Reshape and filter masked rows
        prob_true_X, prob_true_E, prob_pred.X, prob_pred.E = diffusion_utils.mask_distributions(true_X=prob_true.X,
                                                                                                true_E=prob_true.E,
                                                                                                pred_X=prob_pred.X,
                                                                                                pred_E=prob_pred.E,
                                                                                                node_mask=node_mask)
        kl_x = (self.test_X_kl if test else self.val_X_kl)(prob_true.X, torch.log(prob_pred.X))
        kl_e = (self.test_E_kl if test else self.val_E_kl)(prob_true.E, torch.log(prob_pred.E))
        return self.T * (kl_x + kl_e)

    def reconstruction_logp(self, t, X, E, y, node_mask):
        # Compute noise values for t = 0.
        t_zeros = torch.zeros_like(t)
        beta_0 = self.noise_schedule(t_zeros)
        Q0 = self.transition_model.get_Qt(beta_t=beta_0, device=self.device)

        probX0 = X @ Q0.X  # (bs, n, dx_out)
        probE0 = E @ Q0.E.unsqueeze(1)  # (bs, n, n, de_out)

        sampled0 = diffusion_utils.sample_discrete_features(probX=probX0, probE=probE0, node_mask=node_mask)
        sampled0.y = y

        X0 = F.one_hot(sampled0.X, num_classes=self.Xdim_output).float()
        E0 = F.one_hot(sampled0.E, num_classes=self.Edim_output).float()
        y0 = sampled0.y
        assert (X.shape == X0.shape) and (E.shape == E0.shape)

        sampled_0 = utils.PlaceHolder(X=X0, E=E0, y=y0).mask(node_mask)

        # Predictions
        noisy_data = {'X_t': sampled_0.X, 'E_t': sampled_0.E, 'y_t': sampled_0.y, 'node_mask': node_mask,
                      't': torch.zeros(X0.shape[0], 1).type_as(y0)}
        extra_data = self.compute_extra_data(noisy_data)
        pred0 = self.forward(noisy_data, extra_data, node_mask)

        # Normalize predictions
        probX0 = F.softmax(pred0.X, dim=-1)
        probE0 = F.softmax(pred0.E, dim=-1)
        proby0 = F.softmax(pred0.y, dim=-1)

        # Set masked rows to arbitrary values that don't contribute to loss
        probX0[~node_mask] = torch.ones(self.Xdim_output).type_as(probX0)
        probE0[~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2))] = torch.ones(self.Edim_output).type_as(probE0)

        diag_mask = torch.eye(probE0.size(1)).type_as(probE0).bool()
        diag_mask = diag_mask.unsqueeze(0).expand(probE0.size(0), -1, -1)
        probE0[diag_mask] = torch.ones(self.Edim_output).type_as(probE0)

        return utils.PlaceHolder(X=probX0, E=probE0, y=proby0)

    def apply_noise(self, X, E, y, node_mask):
        """ Sample noise and apply it to the data. """

        # Sample a timestep t.
        # When evaluating, the loss for t=0 is computed separately
        lowest_t = 0 if self.training else 1
        t_int = torch.randint(lowest_t, self.T + 1, size=(X.size(0), 1), device=X.device).float()  # (bs, 1)
        s_int = t_int - 1

        t_float = t_int / self.T
        s_float = s_int / self.T

        # beta_t and alpha_s_bar are used for denoising/loss computation
        beta_t = self.noise_schedule(t_normalized=t_float)  # (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s_float)  # (bs, 1)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_float)  # (bs, 1)

        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar,
                                               device=self.device)  # (bs, dx_in, dx_out), (bs, de_in, de_out)
        assert (abs(Qtb.X.sum(dim=2) - 1.) < 1e-4).all(), Qtb.X.sum(dim=2) - 1
        assert (abs(Qtb.E.sum(dim=2) - 1.) < 1e-4).all()

        # Compute transition probabilities
        probX = X @ Qtb.X  # (bs, n, dx_out)
        probE = E @ Qtb.E.unsqueeze(1)  # (bs, n, n, de_out)

        sampled_t = diffusion_utils.sample_discrete_features(probX=probX, probE=probE, node_mask=node_mask)

        X_t = F.one_hot(sampled_t.X, num_classes=self.Xdim_output)
        E_t = F.one_hot(sampled_t.E, num_classes=self.Edim_output)
        assert (X.shape == X_t.shape) and (E.shape == E_t.shape)

        z_t = utils.PlaceHolder(X=X_t, E=E_t, y=y).type_as(X_t).mask(node_mask)

        noisy_data = {'t_int': t_int, 't': t_float, 'beta_t': beta_t, 'alpha_s_bar': alpha_s_bar,
                      'alpha_t_bar': alpha_t_bar, 'X_t': z_t.X, 'E_t': z_t.E, 'y_t': z_t.y, 'node_mask': node_mask}
        return noisy_data

    def compute_val_loss(self, pred, noisy_data, X, E, y, node_mask, test=False):
        """Computes an estimator for the variational lower bound.
           pred: (batch_size, n, total_features)
           noisy_data: dict
           X, E, y : (bs, n, dx),  (bs, n, n, de), (bs, dy)
           node_mask : (bs, n)
           Output: nll (size 1)
       """
        t = noisy_data['t']

        # 1.
        N = node_mask.sum(1).long()
        log_pN = self.node_dist.log_prob(N)

        # 2. The KL between q(z_T | x) and p(z_T) = Uniform(1/num_classes). Should be close to zero.
        kl_prior = self.kl_prior(X, E, node_mask)

        # 3. Diffusion loss
        loss_all_t = self.compute_Lt(X, E, y, pred, noisy_data, node_mask, test)

        # 4. Reconstruction loss
        # Compute L0 term : -log p (X, E, y | z_0) = reconstruction loss
        prob0 = self.reconstruction_logp(t, X, E, y, node_mask)

        loss_term_0 = self.val_X_logp(X * prob0.X.log()) + self.val_E_logp(E * prob0.E.log())

        # Combine terms
        nlls = - log_pN + kl_prior + loss_all_t - loss_term_0
        assert len(nlls.shape) == 1, f'{nlls.shape} has more than only batch dim.'

        # Update NLL metric object and return batch nll
        nll = (self.test_nll if test else self.val_nll)(nlls)  # Average over the batch

        if wandb.run:
            wandb.log({"kl prior": kl_prior.mean(),
                       "Estimator loss terms": loss_all_t.mean(),
                       "log_pn": log_pN.mean(),
                       "loss_term_0": loss_term_0,
                       'batch_test_nll' if test else 'val_nll': nll}, commit=False)
        return nll

    def forward(self, noisy_data, extra_data, node_mask):
        X = torch.cat((noisy_data['X_t'], extra_data.X), dim=2).float()
        E = torch.cat((noisy_data['E_t'], extra_data.E), dim=3).float()
        y = torch.hstack((noisy_data['y_t'], extra_data.y)).float()
        # print("[DEBUG] FORWARD:",noisy_data['y_t'].shape, extra_data.y.shape)
        return self.model(X, E, y, node_mask)

    #@torch.no_grad()
    def sample_batch(self, batch_id: int, batch_size: int, keep_chain: int, number_chain_steps: int,
                     save_final: int, num_nodes=None, y_cond=None, cond=None):
        """
        :param batch_id: int
        :param batch_size: int
        :param num_nodes: int, <int>tensor (batch_size) (optional) for specifying number of nodes
        :param save_final: int: number of predictions to save to file
        :param keep_chain: int: number of chains to save to file
        :param keep_chain_steps: number of timesteps to save for each chain
        :return: molecule_list. Each element of this list is a tuple (atom_types, charges, positions)
        """
        if num_nodes is None:
            n_nodes = self.node_dist.sample_n(batch_size, self.device)
        elif type(num_nodes) == int:
            n_nodes = num_nodes * torch.ones(batch_size, device=self.device, dtype=torch.int)
        else:
            assert isinstance(num_nodes, torch.Tensor)
            n_nodes = num_nodes
        n_max = torch.max(n_nodes).item()
        # Build the masks
        arange = torch.arange(n_max, device=self.device).unsqueeze(0).expand(batch_size, -1)
        node_mask = arange < n_nodes.unsqueeze(1)
        # Sample noise  -- z has size (n_samples, n_nodes, n_features)
        z_T = diffusion_utils.sample_discrete_feature_noise(limit_dist=self.limit_dist, node_mask=node_mask)
        X, E, y = z_T.X, z_T.E, z_T.y

        if y_cond is not None:
            # 确保 y_cond 的 shape 为 (batch_size, y_dim) 或可以广播到这个形状
            print("[DEBUG] y_cond is not none")
            y = y_cond
        else:
            print("[DEBUG] y_cond is none")

        assert (E == torch.transpose(E, 1, 2)).all()
        assert number_chain_steps < self.T
        chain_X_size = torch.Size((number_chain_steps, keep_chain, X.size(1)))
        chain_E_size = torch.Size((number_chain_steps, keep_chain, E.size(1), E.size(2)))
        rchain_X_size = torch.Size((number_chain_steps, batch_size, X.size(1)))
        rchain_E_size = torch.Size((number_chain_steps, batch_size, E.size(1), E.size(2)))

        chain_X = torch.zeros(chain_X_size)
        chain_E = torch.zeros(chain_E_size)

        rchain_X = torch.zeros(rchain_X_size)
        rchain_E = torch.zeros(rchain_E_size)

        sampling_step_relaxed_homos, sampling_step_strict_homos = [], []

        self.homo_state = {
            # 1. 子图节点列表 (List[int])，初始都空
            'nodes': [[] for _ in range(batch_size)],

            # 2. 对应的子图邻接矩阵，按 nodes[i] 维度 n_i×n_i，初始全零
            'sub_adj': [torch.zeros((0, 0), dtype=torch.bool, device=self.device)
                        for _ in range(batch_size)],

            # 3. 上一次 strict_H（prev_strict_H），bool Tensor [bs]
            'prev_strict_H': torch.zeros((batch_size,), dtype=torch.bool, device=self.device),
            'prev_relaxed_H': torch.full((batch_size,), float('inf'), dtype=torch.float, device=self.device)
        }

        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.

        for s_int in reversed(range(0, self.T)):
            #stime = time.time()
            #print(s_int)
            s_array = s_int * torch.ones((batch_size, 1)).type_as(y)
            t_array = s_array + 1
            s_norm = s_array / self.T
            t_norm = t_array / self.T

            # Sample z_s
            sampled_s, discrete_sampled_s, cur_relaxed_homos, cur_strict_homos = self.sample_p_zs_given_zt(s_norm, t_norm, X, E, y, node_mask, cond)
            X, E, _ = sampled_s.X, sampled_s.E, sampled_s.y

            if cur_relaxed_homos != None:
                sampling_step_relaxed_homos.append(cur_relaxed_homos)
                sampling_step_strict_homos.append(cur_strict_homos)

            # Save the first keep_chain graphs
            #               30   *      50             //   50
            write_index = (s_int * number_chain_steps) // self.T
            chain_X[write_index] = discrete_sampled_s.X[:keep_chain]
            chain_E[write_index] = discrete_sampled_s.E[:keep_chain]

            rchain_X[write_index] = discrete_sampled_s.X[:]
            rchain_E[write_index] = discrete_sampled_s.E[:]
            #etime = time.time()
            #print(f"step {s_int},time:{etime-stime}")

        # Sample
        sampled_s = sampled_s.mask(node_mask, collapse=True)
        X, E, _ = sampled_s.X, sampled_s.E, sampled_s.y
        #print(f"[DEBUG] sampling time: {etime-stime}")

        #relaxed_tensor = torch.stack(sampling_step_relaxed_homos, dim=0).cpu()
        #strict_tensor = torch.stack(sampling_step_strict_homos, dim=0).cpu()

        #torch.save(
        #    {'relaxed': relaxed_tensor, 'strict': strict_tensor},
        #    'homo_tracking.pt'
        #)

        # Prepare the chain for saving
        if keep_chain > 0:
            final_X_chain = X[:keep_chain]
            final_E_chain = E[:keep_chain]

            chain_X[0] = final_X_chain  # Overwrite last frame with the resulting X, E
            chain_E[0] = final_E_chain

            chain_X = diffusion_utils.reverse_tensor(chain_X)
            chain_E = diffusion_utils.reverse_tensor(chain_E)

            rchain_X = diffusion_utils.reverse_tensor(rchain_X)
            rchain_E = diffusion_utils.reverse_tensor(rchain_E)

            # Repeat last frame to see final sample better
            chain_X = torch.cat([chain_X, chain_X[-1:].repeat(10, 1, 1)], dim=0)
            chain_E = torch.cat([chain_E, chain_E[-1:].repeat(10, 1, 1, 1)], dim=0)
            assert chain_X.size(0) == (number_chain_steps + 10)

        molecule_list = []
        for i in range(batch_size):
            n = n_nodes[i]
            atom_types = X[i, :n].cpu()
            edge_types = E[i, :n, :n].cpu()
            molecule_list.append([atom_types, edge_types])

        # Visualize chains
        if self.visualization_tools is not None:
            self.print('Visualizing chains...')
            current_path = os.getcwd()
            num_molecules = chain_X.size(1)  # number of molecules
            for i in range(num_molecules):
                result_path = os.path.join(current_path, f'chains/{self.cfg.general.name}/'
                                                         f'epoch{self.current_epoch}/'
                                                         f'chains/molecule_{batch_id + i}')
                if not os.path.exists(result_path):
                    os.makedirs(result_path)
                    _ = self.visualization_tools.visualize_chain(result_path,
                                                                 chain_X[:, i, :].numpy(),
                                                                 chain_E[:, i, :].numpy())
                self.print('\r{}/{} complete'.format(i + 1, num_molecules), end='', flush=True)
            self.print('\nVisualizing molecules...')

            # Visualize the final molecules
            current_path = os.getcwd()
            result_path = os.path.join(current_path,
                                       f'graphs/{self.name}/epoch{self.current_epoch}_b{batch_id}/')
            self.visualization_tools.visualize(result_path, molecule_list, save_final)
            self.print("Done.")

        return molecule_list, rchain_X, rchain_E


    def cond_fn(self, noisy_data, node_mask, target=None, target_type="path", cid=999):
        with torch.enable_grad():
            # 获取 E_t，并取出第二个通道作为独立变量
            E = noisy_data['E_t']
            E_v_ind = E[..., 1]
            bs, n, _, _ = E.shape

            # 将 E_v_ind 重复复制后设置 requires_grad=True
            E_v_rep = E_v_ind.repeat_interleave(self.condi_delta, dim=0).clone().detach().float().requires_grad_(True)

            if "path" in target_type:
                src_idx_list = []
                tgt_idx_list = []
                dist_list = []
                for i in range(bs):
                    # 重构成矩阵，并展平为长度 n*n 的一维向量
                    target_mat = target[i]  # shape: (n, n)
                    target_flat = target_mat.view(-1)  # shape: (n*n,)

                    # 构造基于索引的扰动项，确保相同值之间排序稳定
                    indices = torch.arange(target_flat.size(0), device=target_flat.device,
                                           dtype=target_flat.dtype)
                    stable_target = target_flat + indices / target_flat.size(0)

                    # 构造上三角部分的 mask：仅保留 row < col 的位置
                    rows = indices // n
                    cols = indices % n
                    upper_mask = rows < cols

                    # 筛选出上三角部分的 stable_target 以及其对应的原始展平索引
                    stable_target_upper = stable_target[upper_mask]
                    valid_indices = torch.nonzero(upper_mask, as_tuple=False).view(-1)

                    # 这里假设上三角部分的元素足够多，否则需要做容错处理
                    # 例如这里取的是第800和第1000大的元素（对应于 topk 中的索引799和999）
                    topk = torch.topk(stable_target_upper, cid+1, largest=True)
                    # 固定取第800和第1000个采样
                    sampled_relative_indices = torch.stack([topk.indices[cid]])
                    # 将相对索引映射为原始展平索引
                    sampled_indices = valid_indices[sampled_relative_indices]

                    # 根据展平索引计算二维位置
                    src_idx = sampled_indices // n
                    tgt_idx = sampled_indices % n

                    src_idx_list.append(src_idx)
                    tgt_idx_list.append(tgt_idx)
                    dist_list.append(target_flat[sampled_indices])

                src_idx = torch.stack(src_idx_list, dim=0).to(self.device)
                tgt_idx = torch.stack(tgt_idx_list, dim=0).to(self.device)
                dist = torch.stack(dist_list, dim=0).to(self.device)
                #delta = 2  # 固定采样数量为2

                src_idx_flat = src_idx.view(-1)
                tgt_idx_flat = tgt_idx.view(-1)
                target_flat = dist.view(-1)



                # 前向传播，调用 guidance_model
                pred = self.guidance_model(E_v_rep, src_idx_flat, tgt_idx_flat)

            else:
                pred = self.guidance_model(E_v_rep, node_mask)
                #print(pred.shape, target.shape)
                target_flat = target
            # 计算损失并反向传播
            if pred.shape!=target_flat.shape:
                target_flat = target_flat.reshape(pred.shape)
            loss = F.mse_loss(pred, target_flat)
            loss.backward()

            # 计算并返回最终梯度
            grad_rep = E_v_rep.grad.detach()
            grad_rep = grad_rep.view(bs, self.condi_delta, n, n)
            final_grad = grad_rep.mean(dim=1)
            #final_grad = 0.5 * (final_grad + final_grad.transpose(1, 2))
            return final_grad

    def condition_e(self, prob_E, grad_E, MODE="+", la=1,method="avg"):
        if method == "avg":
            grad_E = (grad_E + grad_E.transpose(1, 2)) / 2
        else:
            grad_E = torch.triu(grad_E) + torch.triu(grad_E, diagonal=1).transpose(1, 2)

        if "top" in MODE:
            bs, n, _ = grad_E.shape
            flat_grad_E = grad_E.view(bs, -1)
            top_vals, top_indices = flat_grad_E.abs().topk(self.sampling_multiplier * self.condi_delta, dim=1)
            mask = torch.zeros_like(flat_grad_E, dtype=torch.bool)
            mask.scatter_(1, top_indices, True)
            flat_grad_E_filtered = flat_grad_E * mask.float()
            grad_E = flat_grad_E_filtered.view(bs, n, n)

        if "*" in MODE:
            grad_neg = 1 / (1 + torch.exp(-grad_E))
            grad_pos = 1 / (1 + torch.exp(grad_E))
            diff = torch.abs(grad_neg - grad_pos)
            # 构造一个条件掩码：当差值小于等于 0.5 时，对应位置为 True
            condition = diff <= 0.8
            grad_neg[condition] = 1/la
            grad_pos[condition] = 1/la
        # for +,grad->0, guidance->0
        elif "+" in MODE:
            grad_pos = torch.zeros_like(grad_E)
            grad_neg = torch.zeros_like(grad_E)

            pos_mask = grad_E < -1000000
            grad_pos[pos_mask] = 2 * (torch.sigmoid(-0.5 * grad_E[pos_mask]) - 0.5)

            neg_mask = grad_E > 1000000
            grad_neg[neg_mask] = 2 * (torch.sigmoid(0.5 * grad_E[neg_mask]) - 0.5)

        grad_e_s = torch.stack((grad_neg, grad_pos), dim=-1)
        if MODE == "+" or MODE == "top+":
            prob_E_unnormalized = prob_E + la * grad_e_s
            prob_E_unnormalized[torch.sum(prob_E_unnormalized, dim=-1) == 0] = 1e-7
            prob_E = prob_E_unnormalized / torch.sum(prob_E_unnormalized, dim=-1, keepdim=True)
        elif MODE == "*":
            grad_e_s = torch.softmax(la * grad_e_s, dim=-1)
            #grad_e_s = la * grad_e_s
            prob_E_unnormalized = grad_e_s * prob_E
            prob_E_unnormalized[torch.sum(prob_E_unnormalized, dim=-1) == 0] = 1e-8
            prob_E = prob_E_unnormalized / torch.sum(prob_E_unnormalized, dim=-1, keepdim=True)
        return prob_E


    def sample_p_zs_given_zt(self, s, t, X_t, E_t, y_t, node_mask, cond=None):
        """Samples from zs ~ p(zs | zt). Only used during sampling.
           if last_step, return the graph prediction as well"""
        bs, n, dxs = X_t.shape
        beta_t = self.noise_schedule(t_normalized=t)  # (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t)

        # Retrieve transitions matrix
        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.device)
        Qsb = self.transition_model.get_Qt_bar(alpha_s_bar, self.device)
        Qt = self.transition_model.get_Qt(beta_t, self.device)

        relaxed_homos, strict_homos = None, None

        # Neural net predictions
        noisy_data = {'X_t': X_t, 'E_t': E_t, 'y_t': y_t, 't': t, 'node_mask': node_mask}

        # target = cond_fn(cond, self.device)  # shape [bs]

        # self.flag == 0, 无条件生成阶段
            # 𝐻(∙)实时监控生成，返回0则没达到条件
        #relaxed_flag, strict_flag = condition_utils.condition_relaxed_H(noisy_data, cond, self.condition_target, self.device)
        #if (1-s[0][0]) >= self.condition_phase_s and  (1-s[0][0]) < self.condition_phase_e and self.enable_condition:
        #    # conditional generation
        #    with torch.enable_grad():
        #        grad_e = self.cond_fn(noisy_data, node_mask, cond)

        #    grad_x, grad_e = self.cond_fn(noisy_data, node_mask, input_properties)
        #    prob_E = self.condition_e(prob_E, grad_e, self.condition_mode, self.condition_weight, "avg")
            #noisy_data_homos = condition_utils.condition_homo_test_gen(grad_e, noisy_data, self.device)
            #relaxed_homos, strict_homos = condition_utils.condition_homo_test_check(noisy_data_homos, target, "path", self.device)

        bs = noisy_data['E_t'].shape[0]

        #  and len(empty_idx)!=0 and int((1 - s[0][0])*1000+1)%10==0
        if (1 - s[0][0]) >= self.homo_condition_phase_s and (1 - s[0][0]) < self.homo_condition_phase_e and self.enable_condition and int((1 - s[0][0])*1000+1)%10==0:
            #print(self.homo_state['prev_relaxed_H'])
            # print(self.homo_state['prev_strict_H'])

            with torch.enable_grad():
                grad_e = self.cond_fn(noisy_data, node_mask, cond, target_type=condi_config.condition_target)

            # STEP 1: conditional generation with persistent-homology ----- graph candidates
            homo_list = condition_utils.condition_homo_test_gen(
                grad_e, noisy_data, self.device
            )
            # Insert the original graph to unify the sampling procedure,
            #     ensuring that it is selected when no persistent-homology candidate satisfies the condition
            homo_list.insert(0, noisy_data)

            #  STEP 2: Compute the weight for each graph and return the final selected conditional graph ----- weight computation
            noisy_prime = condition_utils.condition_homo_check(
                homo_list, cond, condi_config.condition_target, self.device
            )

            #  STEP 3: Write the selected graph back into the generation pipeline ----- write back
            for idx, _ in enumerate(noisy_prime):
                    # noisy_prime[idx] is in the form {'E_t': Tensor[1, n, n, 2], 'X_t': Tensor[1, ...], ...}
                    prime = noisy_prime[idx]

                    for key, val in prime.items():
                        # val.squeeze(0) makes the shape match noisy_data[key][i]
                        noisy_data[key][idx] = val.squeeze(0)

        extra_data = self.compute_extra_data(noisy_data)
        pred = self.forward(noisy_data, extra_data, node_mask)

        # Normalize predictions
        pred_X = F.softmax(pred.X, dim=-1)  # bs, n, d0
        pred_E = F.softmax(pred.E, dim=-1)  # bs, n, n, d0

        p_s_and_t_given_0_X = diffusion_utils.compute_batched_over0_posterior_distribution(X_t=X_t,
                                                                                           Qt=Qt.X,
                                                                                           Qsb=Qsb.X,
                                                                                           Qtb=Qtb.X)

        p_s_and_t_given_0_E = diffusion_utils.compute_batched_over0_posterior_distribution(X_t=E_t,
                                                                                           Qt=Qt.E,
                                                                                           Qsb=Qsb.E,
                                                                                           Qtb=Qtb.E)
        # Dim of these two tensors: bs, N, d0, d_t-1
        weighted_X = pred_X.unsqueeze(-1) * p_s_and_t_given_0_X  # bs, n, d0, ad_t-1
        unnormalized_prob_X = weighted_X.sum(dim=2)  # bs, n, d_t-1
        unnormalized_prob_X[torch.sum(unnormalized_prob_X, dim=-1) == 0] = 1e-5
        prob_X = unnormalized_prob_X / torch.sum(unnormalized_prob_X, dim=-1, keepdim=True)  # bs, n, d_t-1

        pred_E = pred_E.reshape((bs, -1, pred_E.shape[-1]))
        weighted_E = pred_E.unsqueeze(-1) * p_s_and_t_given_0_E  # bs, N, d0, d_t-1
        unnormalized_prob_E = weighted_E.sum(dim=-2)
        unnormalized_prob_E[torch.sum(unnormalized_prob_E, dim=-1) == 0] = 1e-5
        prob_E = unnormalized_prob_E / torch.sum(unnormalized_prob_E, dim=-1, keepdim=True)
        prob_E = prob_E.reshape(bs, n, n, pred_E.shape[-1])

        if (1-s[0][0]) >= self.all_condition_phase_s and  (1-s[0][0]) < self.all_condition_phase_e and self.enable_condition:
            # conditional generation
            print(1)
            with torch.enable_grad():
                grad_e = self.cond_fn(noisy_data, node_mask, cond, target_type=condi_config.condition_target)

            # grad_x, grad_e = self.cond_fn(noisy_data, node_mask, input_properties)
            # prob_E = self.condition_e(prob_E, grad_e, self.condition_mode, self.condition_weight,"avg")
        #else:
            # with torch.enable_grad():
            #     grad_e = self.cond_fn(noisy_data, node_mask, cond)
            #
            # grad_x, grad_e = self.cond_fn(noisy_data, node_mask, input_properties)
            # prob_E = self.condition_e(prob_E, grad_e, "*", self.condition_weight,"upper")

        assert ((prob_X.sum(dim=-1) - 1).abs() < 1e-4).all()
        assert ((prob_E.sum(dim=-1) - 1).abs() < 1e-4).all()

        sampled_s = diffusion_utils.sample_discrete_features(prob_X, prob_E, node_mask=node_mask)

        X_s = F.one_hot(sampled_s.X, num_classes=self.Xdim_output).float()
        E_s = F.one_hot(sampled_s.E, num_classes=self.Edim_output).float()

        assert (E_s == torch.transpose(E_s, 1, 2)).all()
        assert (X_t.shape == X_s.shape) and (E_t.shape == E_s.shape)

        out_one_hot = utils.PlaceHolder(X=X_s, E=E_s, y=torch.zeros(y_t.shape[0], 0))
        out_discrete = utils.PlaceHolder(X=X_s, E=E_s, y=torch.zeros(y_t.shape[0], 0))

        return (out_one_hot.mask(node_mask).type_as(y_t), out_discrete.mask(node_mask, collapse=True).type_as(y_t),
                relaxed_homos, strict_homos)

    def compute_extra_data(self, noisy_data):
        """ At every training step (after adding noise) and step in sampling, compute extra information and append to
            the network input. """

        extra_features = self.extra_features(noisy_data)
        extra_molecular_features = self.domain_features(noisy_data)

        extra_X = torch.cat((extra_features.X, extra_molecular_features.X), dim=-1)
        extra_E = torch.cat((extra_features.E, extra_molecular_features.E), dim=-1)
        extra_y = torch.cat((extra_features.y, extra_molecular_features.y), dim=-1)

        t = noisy_data['t']
        extra_y = torch.cat((extra_y, t), dim=1)

        return utils.PlaceHolder(X=extra_X, E=extra_E, y=extra_y)



