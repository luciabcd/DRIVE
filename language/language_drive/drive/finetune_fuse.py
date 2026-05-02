import os
import sys
import logging
import torch
import numpy as np
import argparse
import json
import wandb
import torch.distributed as dist
from datetime import datetime
from transformers import TrainingArguments, Trainer
from accelerate import Accelerator
from torch.utils.data.dataloader import default_collate 
from torch.nn.utils.rnn import pad_sequence 
import torch.nn.functional as F
from transformers import TrainerCallback
from collections import deque


from models import create_model_tokenizer_it, create_peft_model_it, IGNORE_INDEX
from utils.data_utils import load_and_preprocess_it
from utils.parsing_utils import str_to_bool


logger = logging.getLogger(__name__)



def init_distributed():
    if dist.is_initialized():
        return
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)

def create_run_directory(args):
    schedule_tag = args.lambda_schedule
    timestamp = datetime.now().strftime("%m%d_%H%M")
    run_name = f"{args.model.split('/')[-1]}_{schedule_tag}_{timestamp}"
    run_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=4)
    return run_dir



class IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, original_dataset):
        self.dataset = original_dataset
    def __len__(self): return len(self.dataset)
    def __getitem__(self, idx):
        item = self.dataset[idx]
        item['index'] = idx
        return item

def SimpleDataCollatorWithIndex(features):

    if "index" in features[0]:
        indices = [f.pop("index") for f in features]
    else:
        raise KeyError("DataCollator: 'index' key is missing!")

    try:
        input_ids_list = [f.pop("input_ids") for f in features]
        labels_list = [f.pop("labels") for f in features]
    except KeyError as e:
        raise KeyError(f"Missing required key {e} in dataset item.")

    input_ids_tensors = [torch.tensor(x, dtype=torch.long) if isinstance(x, list) else x for x in input_ids_list]
    labels_tensors = [torch.tensor(x, dtype=torch.long) if isinstance(x, list) else x for x in labels_list]
    
    PAD_ID = 0 
    IGNORE_INDEX_ID = IGNORE_INDEX 

    input_ids_tensor = pad_sequence(input_ids_tensors, batch_first=True, padding_value=PAD_ID)
    labels_tensor = pad_sequence(labels_tensors, batch_first=True, padding_value=IGNORE_INDEX_ID)
    attention_mask_tensor = (input_ids_tensor != PAD_ID).long()

    batch = {
        "input_ids": input_ids_tensor,
        "labels": labels_tensor,
        "attention_mask": attention_mask_tensor,
        "index": torch.tensor(indices, dtype=torch.long)
    }
    
    return batch

def calculate_iterative_weights_drive(c_scores, z_batch, eta, tau, projection_dim=128, T_max=5):


    z_batch = z_batch.detach().float()
    c_scores = c_scores.detach().float()
    
    device = z_batch.device
    B, D = z_batch.shape
    

    z_batch = F.normalize(z_batch, p=2, dim=1)


    if D > projection_dim:
        R = torch.randn(D, projection_dim, device=device) / (projection_dim ** 0.5)
        z_proj = torch.mm(z_batch, R) # [B, k]
    else:
        z_proj = z_batch
            
    z_proj = z_proj - z_proj.mean(dim=0, keepdim=True)

    w = torch.ones(B, device=device) / B
    
    epsilon_val = 1e-5
    k = z_proj.shape[1]
    eye_k = torch.eye(k, device=device) * epsilon_val
    for t in range(T_max):
        z_weighted = z_proj * torch.sqrt(w.unsqueeze(1) + 1e-10) 
        Sigma = torch.mm(z_weighted.t(), z_weighted) + eye_k # [k, k]
        try:
            L_factor = torch.linalg.cholesky(Sigma)
            gamma_z = torch.cholesky_solve(z_proj.t(), L_factor) 
            L_scores = (z_proj.t() * gamma_z).sum(dim=0) # [B]
        except RuntimeError:
            L_scores = torch.zeros(B, device=device)
        logits = (c_scores + eta * L_scores) / tau
        logits = torch.clamp(logits, min=-20.0, max=20.0)
        
        w_new = torch.exp(logits)
        w_sum = w_new.sum()
        if w_sum < 1e-9:
             w_new = torch.ones(B, device=device) / B
        else:
             w_new = w_new / w_sum
        
        w = w_new

    return w 

class DriveDynamicTrainer(Trainer):
    def __init__(self, drive_cache_dir, drive_args, **kwargs):
        super().__init__(**kwargs)
        self.drive_args = drive_args
        self.device_tensor = torch.device(self.args.device) if self.args.device else torch.device("cuda")
        
        if self.is_world_process_zero():
            logger.info(f"Trainer: Loading DRIVE cache from {drive_cache_dir}...")
        self.ref_losses = torch.tensor(np.load(os.path.join(drive_cache_dir, "ref_losses.npy"))).to(self.device_tensor)
        self.ref_features = torch.tensor(np.load(os.path.join(drive_cache_dir, "ref_features.npy"))).to(self.device_tensor)
        self.bank_size = 128
        self.z_bank = [] 
        self.c_bank = [] 

    def update_bank(self, z_curr, c_curr):

        z_detach = z_curr.detach().float()
        c_detach = c_curr.detach().float()

        for i in range(z_curr.shape[0]):
            self.z_bank.append(z_detach[i])
            self.c_bank.append(c_detach[i])
        
        while len(self.z_bank) > self.bank_size:
            self.z_bank.pop(0)
            self.c_bank.pop(0)

    def get_augmented_batch(self, z_curr, c_curr):

        if len(self.z_bank) < 10: 
            return z_curr, c_curr
        
        z_history = torch.stack(self.z_bank).to(z_curr.device)
        c_history = torch.stack(self.c_bank).to(c_curr.device)

        z_mix = torch.cat([z_curr, z_history], dim=0)
        c_mix = torch.cat([c_curr, c_history], dim=0)
        
        return z_mix, c_mix

    def get_ref_data(self, indices):
        indices = indices.to(self.device_tensor)
        a_batch = self.ref_losses[indices]
        z_batch = self.ref_features[indices]
        return a_batch, z_batch

    def get_dynamic_lambda(self):
        total_steps = self.state.max_steps
        current_step = self.state.global_step
        if total_steps <= 0: progress = 0.0
        else: progress = current_step / total_steps
        progress = min(max(progress, 0.0), 1.0)

        mode = self.drive_args.lambda_schedule
        if mode == "linear_decay": lam = 1.0 - progress
        elif mode == "linear_increase": lam = progress
        elif mode == "fixed": lam = self.drive_args.lambda_weight
        else: raise ValueError(f"Unknown schedule: {mode}")
        return lam

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):

        if "index" in inputs:
            indices = inputs.pop("index")
        else:
            return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
        outputs = model(**inputs)
        
        logits = outputs.get("logits")
        labels = inputs.get("labels")
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=IGNORE_INDEX)
        token_losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        token_losses = token_losses.view(shift_labels.shape)
        
        valid_token_mask = shift_labels.ne(IGNORE_INDEX).float()
        valid_token_count = valid_token_mask.sum(dim=1)
        t_batch = (token_losses * valid_token_mask).sum(dim=1) / (valid_token_count + 1e-9)
        t_batch = torch.nan_to_num(t_batch, nan=0.0, posinf=0.0)

        ref_loss_vals, ref_features = self.get_ref_data(indices)
        if torch.isnan(ref_loss_vals).any(): 
            ref_loss_vals = torch.nan_to_num(ref_loss_vals, nan=5.0)

        a_batch = -1.0 * ref_loss_vals 

        current_lambda = self.get_dynamic_lambda()

        c_scores = current_lambda * a_batch + (1 - current_lambda) * t_batch.detach()
        z_curr_detach = ref_features.detach()
        c_curr_detach = c_scores.detach()
        
        z_mix, c_mix = self.get_augmented_batch(z_curr_detach, c_curr_detach)

        weights_mix = calculate_iterative_weights_drive(
            c_scores=c_mix,   
            z_batch=z_mix,    
            eta=self.drive_args.drive_gamma, 
            tau=self.drive_args.drive_tau,   
            projection_dim=128,
            T_max=5
        )

        current_bs = t_batch.shape[0]
        weights = weights_mix[:current_bs]

        self.update_bank(z_curr_detach, c_curr_detach)

        weights = (weights / (weights.sum() + 1e-9)) * current_bs
        
        weights = weights.detach() 

        loss_per_sample = weights * t_batch
        weighted_loss = loss_per_sample.mean()
        
        if torch.isnan(weighted_loss) or torch.isinf(weighted_loss):
            logger.error("🚨 CRITICAL: Final Loss is NaN/Inf! Fallback to unweighted loss.")
            weighted_loss = t_batch.mean()

        return (weighted_loss, outputs) if return_outputs else weighted_loss

class FileLoggerCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_world_process_zero and logs is not None:
            log_str = " | ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" 
                                  for k, v in logs.items() if k not in ["epoch"]])
            if "epoch" in logs:
                logger.info(f"Epoch: {logs['epoch']:.2f} | {log_str}")
            else:
                logger.info(log_str)

def finetune(args):
    init_distributed()
    accelerator = Accelerator()
    
    if accelerator.is_main_process:
        run_dir = create_run_directory(args)
    else:
        run_dir = None
    
    if dist.is_initialized():
        object_list = [run_dir]
        dist.broadcast_object_list(object_list, src=0)
        run_dir = object_list[0]
        
    if accelerator.is_main_process:
        log_file_path = os.path.join(run_dir, "logs", "training.log")
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(log_file_path, mode="w")
            ]
        )
        global logger
        logger = logging.getLogger(__name__)
        logger.info(f"Logging configured. Logs saved to: {log_file_path}")
    else:
        logging.basicConfig(level=logging.ERROR, handlers=[logging.StreamHandler(sys.stdout)])
        logger = logging.getLogger(__name__)

    if not args.no_wandb and accelerator.is_main_process:
        wandb.init(project=args.project_name, config=args, name=os.path.basename(run_dir))

    training_args = TrainingArguments(
        output_dir=os.path.join(run_dir, "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        save_strategy="epoch",
        logging_steps=5,
        logging_dir=os.path.join(run_dir, "logs"),
        remove_unused_columns=False,
        bf16=True,
        optim=args.optim,
        report_to="wandb" if not args.no_wandb else "none",
        dataloader_num_workers=4,
        ddp_find_unused_parameters=False,
    )

    logger.info("Creating model and tokenizer...")
    model, tokenizer = create_model_tokenizer_it(args)
    model = model.to(dtype=torch.bfloat16)
    
    if args.finetune_type == "lora":
        model, _ = create_peft_model_it(model, args)
    
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    logger.info("Loading dataset...")
    raw_dataset = load_and_preprocess_it(tokenizer=tokenizer, args=args)
    train_dataset = IndexedDataset(raw_dataset)
    
    trainer = DriveDynamicTrainer(
        drive_cache_dir=args.cache_dir,
        drive_args=args,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=SimpleDataCollatorWithIndex,
        callbacks=[FileLoggerCallback] 
    )

    model.config.use_cache = False
    
    logger.info("Starting training...")
    trainer.train()
    
    if accelerator.is_main_process:
        save_path = os.path.join(run_dir, "final_model")
        trainer.save_model(save_path)
        tokenizer.save_pretrained(os.path.join(run_dir, "tokenizer"))
        logger.info(f"🎉 Training finished. Model saved to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Path Arguments
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="experiments")
    
    # DRIVE Arguments
    parser.add_argument("--lambda_schedule", type=str, default="fixed")
    parser.add_argument("--lambda_weight", type=float, default=0.5)
    parser.add_argument("--drive_gamma", type=float, default=0.1, help="Diversity strength (eta)")
    parser.add_argument("--drive_tau", type=float, default=1.0, help="Entropy strength (tau)")
    
    # Training Arguments
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--dataset_field", type=str, nargs="+", default=["query", "response"])
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--finetune_type", type=str, default="lora")
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--gradient_checkpointing", type=str, default="true")
    parser.add_argument("--optim", type=str, default="adamw_torch")
    parser.add_argument("--no_wandb", type=str, default="false")
    parser.add_argument("--project_name", type=str, default="drive-dynamic")
    parser.add_argument("--local_rank", type=int, default=-1)

    args = parser.parse_args()
    args.no_wandb = str_to_bool(args.no_wandb)
    args.gradient_checkpointing = str_to_bool(args.gradient_checkpointing)
    
    finetune(args)