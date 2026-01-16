"""
Trainer for Concerto-LAQ model.

Handles training loop, logging, checkpointing, and distributed training
for the Concerto-based Latent Action Quantization model.
"""

from math import sqrt
from random import choice
from pathlib import Path
from shutil import rmtree
from typing import Optional, Callable

import wandb
from beartype import beartype

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.utils import make_grid, save_image

from accelerate import Accelerator, DistributedDataParallelKwargs

from einops import rearrange

from laq_model.concerto_laq import ConcertoLAQ
from laq_model.concerto_data import ConcertoVideoDataset, PrecomputedFeatureDataset, collate_fn
from laq_model.optimizer import get_optimizer


def exists(val):
    return val is not None


def noop(*args, **kwargs):
    pass


def cycle(dl):
    while True:
        for data in dl:
            yield data


def accum_log(log, new_logs):
    for key, new_value in new_logs.items():
        old_value = log.get(key, 0.)
        log[key] = old_value + new_value
    return log


@beartype
class ConcertoLAQTrainer(nn.Module):
    """
    Trainer for Concerto-LAQ model.
    
    Supports:
    - Training on raw video with on-the-fly Concerto extraction
    - Training on pre-computed Concerto features (faster)
    - Distributed training with Accelerate
    - WandB logging
    - Gradient accumulation
    - EMA model
    
    Args:
        model: ConcertoLAQ model instance
        folder: Path to training data
        num_train_steps: Total training steps
        batch_size: Training batch size
        lr: Learning rate
        grad_accum_every: Gradient accumulation steps
        wd: Weight decay
        max_grad_norm: Maximum gradient norm for clipping
        save_results_every: Frequency for saving visualizations
        save_model_every: Frequency for saving checkpoints
        results_folder: Path to save results
        use_ema: Whether to use Exponential Moving Average
        use_precomputed_features: Whether to use pre-computed Concerto features
        feature_cache_dir: Directory with pre-computed features
        offset: Frame offset for training pairs
    """
    
    def __init__(
        self,
        model: ConcertoLAQ,
        *,
        folder: str,
        num_train_steps: int,
        batch_size: int,
        lr: float = 3e-4,
        grad_accum_every: int = 1,
        wd: float = 0.,
        max_grad_norm: float = 0.5,
        save_results_every: int = 50,
        save_model_every: int = 5000,
        results_folder: str = './results_concerto',
        use_ema: bool = False,
        ema_update_after_step: int = 0,
        ema_update_every: int = 1,
        use_precomputed_features: bool = False,
        feature_cache_dir: Optional[str] = None,
        offset: int = 5,
        accelerate_kwargs: dict = {},
        wandb_project: str = 'concerto_laq',
    ):
        super().__init__()
        
        # Distributed training setup
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(**accelerate_kwargs, kwargs_handlers=[ddp_kwargs])
        
        self.model = model
        self.results_folder_str = results_folder
        self.lr = lr
        self.wandb_project = wandb_project
        
        # EMA model
        self.use_ema = use_ema
        if self.is_main and use_ema:
            from ema_pytorch import EMA
            self.ema_model = EMA(
                model, 
                update_after_step=ema_update_after_step, 
                update_every=ema_update_every
            )
        
        self.register_buffer('steps', torch.Tensor([0]))
        
        self.num_train_steps = num_train_steps
        self.batch_size = batch_size
        self.grad_accum_every = grad_accum_every
        self.max_grad_norm = max_grad_norm
        self.save_results_every = save_results_every
        self.save_model_every = save_model_every
        
        # Optimizer (only trainable params, not frozen Concerto)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        self.optim = get_optimizer(trainable_params, lr=lr, wd=wd)
        
        # Dataset
        self.use_precomputed_features = use_precomputed_features
        if use_precomputed_features and feature_cache_dir:
            self.ds = PrecomputedFeatureDataset(
                feature_dir=feature_cache_dir,
                offset=offset,
            )
        else:
            self.ds = ConcertoVideoDataset(
                folder=folder,
                image_size=model.image_size[0],
                offset=offset,
                feature_cache_dir=feature_cache_dir,
                use_depth=model.depth_estimator is not None,
            )
        
        self.valid_ds = self.ds
        
        # DataLoaders
        self.dl = DataLoader(
            self.ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=2,
            collate_fn=collate_fn,
        )
        
        self.valid_dl = DataLoader(
            self.valid_ds,
            batch_size=batch_size,
            num_workers=4,
            collate_fn=collate_fn,
        )
        
        # Prepare for distributed training
        self.model, self.optim, self.dl = self.accelerator.prepare(
            self.model, self.optim, self.dl
        )
        
        self.dl_iter = cycle(self.dl)
        self.valid_dl_iter = cycle(self.valid_dl)
        
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(parents=True, exist_ok=True)
    
    def save(self, path):
        """Save checkpoint."""
        if not self.accelerator.is_local_main_process:
            return
        
        pkg = dict(
            model=self.accelerator.get_state_dict(self.model),
            optim=self.optim.state_dict(),
            steps=self.steps.item()
        )
        
        torch.save(pkg, path)
    
    def load(self, path):
        """Load checkpoint."""
        path = Path(path)
        assert path.exists()
        pkg = torch.load(path)
        
        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(pkg['model'])
        self.optim.load_state_dict(pkg['optim'])
        self.steps = torch.tensor([pkg.get('steps', 0)])
    
    def print(self, msg):
        self.accelerator.print(msg)
    
    @property
    def device(self):
        return self.accelerator.device
    
    @property
    def is_main(self):
        return self.accelerator.is_main_process
    
    @property
    def is_local_main(self):
        return self.accelerator.is_local_main_process
    
    def train_step(self):
        """Single training step."""
        device = self.device
        steps = int(self.steps.item())
        
        self.model.train()
        logs = {}
        
        for _ in range(self.grad_accum_every):
            batch = next(self.dl_iter)
            
            # Prepare inputs based on what's available
            kwargs = {'step': steps}
            
            if 'features' in batch:
                kwargs['features'] = batch['features'].to(device)
            else:
                kwargs['video'] = batch['frames'].to(device)
                if 'depth' in batch:
                    kwargs['depth'] = batch['depth'].to(device)
            
            # Forward pass
            loss, num_unique_indices = self.model(**kwargs)
            
            # Backward pass
            self.accelerator.backward(loss / self.grad_accum_every)
            
            accum_log(logs, {
                'loss': loss.item() / self.grad_accum_every,
                'num_unique_indices': num_unique_indices,
            })
        
        # Gradient clipping
        if exists(self.max_grad_norm):
            self.accelerator.clip_grad_norm_(
                self.model.parameters(), 
                self.max_grad_norm
            )
        
        self.optim.step()
        self.optim.zero_grad()
        
        # Logging
        if self.is_main:
            wandb.log(logs)
        
        # EMA update
        if self.is_main and self.use_ema:
            self.ema_model.update()
        
        # Save visualizations
        if self.is_main and not (steps % self.save_results_every):
            self._save_visualizations(steps)
        
        # Save checkpoint
        if self.is_main and not (steps % self.save_model_every):
            self._save_checkpoint(steps)
        
        self.steps += 1
        return logs
    
    def _save_visualizations(self, steps: int):
        """Save visualization of predictions."""
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        models_to_eval = [(unwrapped_model, str(steps))]
        
        if self.use_ema:
            models_to_eval.append((self.ema_model.ema_model, f'{steps}.ema'))
        
        for model, filename in models_to_eval:
            model.eval()
            
            batch = next(self.valid_dl_iter)
            
            # Prepare inputs
            kwargs = {}
            if 'features' in batch:
                kwargs['features'] = batch['features'].to(self.device)
            else:
                kwargs['video'] = batch['frames'].to(self.device)
                if 'depth' in batch:
                    kwargs['depth'] = batch['depth'].to(self.device)
            
            with torch.no_grad():
                # Get action indices
                indices = model(**kwargs, return_only_codebook_ids=True)
                
                # Log codebook usage
                unique_indices = indices.unique()
                wandb.log({
                    f'{filename}_unique_codes': len(unique_indices),
                    f'{filename}_indices_hist': wandb.Histogram(indices.cpu().numpy().flatten()),
                })
            
            self.print(f'{steps}: visualizations saved to {self.results_folder}')
    
    def _save_checkpoint(self, steps: int):
        """Save model checkpoint."""
        state_dict = self.accelerator.unwrap_model(self.model).state_dict()
        model_path = str(self.results_folder / f'concerto_laq.{steps}.pt')
        torch.save(state_dict, model_path)
        
        if self.use_ema:
            ema_state_dict = self.ema_model.state_dict()
            ema_path = str(self.results_folder / f'concerto_laq.{steps}.ema.pt')
            torch.save(ema_state_dict, ema_path)
        
        self.print(f'{steps}: checkpoint saved to {self.results_folder}')
    
    def train(self, log_fn: Callable = noop):
        """Main training loop."""
        device = next(self.model.parameters()).device
        
        if self.accelerator.is_main_process:
            wandb.init(
                project=self.wandb_project,
                name=self.results_folder_str.split('/')[-1],
                config={
                    "learning_rate": self.lr,
                    "batch_size": self.batch_size,
                    "num_train_steps": self.num_train_steps,
                }
            )
        
        while self.steps < self.num_train_steps:
            logs = self.train_step()
            log_fn(logs)
        
        self.print('training complete')
        
        if self.accelerator.is_main_process:
            wandb.finish()
