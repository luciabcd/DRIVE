import os
import torch
import numpy as np
import argparse
import torch.distributed as dist
from tqdm import tqdm
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data.dataloader import default_collate 
from dataclasses import dataclass
from typing import List, Dict, Any


from models import create_model_tokenizer_it
from utils.data_utils import load_and_preprocess_it, IGNORE_INDEX 

def init_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group(backend="nccl", world_size=world_size, rank=rank)
        torch.cuda.set_device(rank)
        return rank, world_size
    else:
        return 0, 1

class IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, original_dataset):
        self.dataset = original_dataset
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        item = self.dataset[idx]
        item['index'] = idx 
        return item


@dataclass
class DataCollatorWithIndexPad:
    tokenizer: Any
    max_seq_length: int
    label_pad_token_id: int = IGNORE_INDEX  

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        if "index" not in features[0]:
            raise KeyError("DataCollator: 'index' key is missing from the dataset item!")
        indices = [f.pop("index") for f in features]
        input_ids_list = [f["input_ids"] for f in features]
        labels_list = [f["labels"] for f in features]
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            if self.tokenizer.eos_token_id is None:
                raise ValueError("Tokenizer has no pad_token_id and no eos_token_id; cannot pad.")
            pad_id = self.tokenizer.eos_token_id

        batch_max = max(len(x) for x in input_ids_list)
        tgt_len = min(batch_max, self.max_seq_length)

        def pad_to_len(seq, pad_value):
            seq = seq[:tgt_len]
            if len(seq) < tgt_len:
                seq = seq + [pad_value] * (tgt_len - len(seq))
            return seq

        input_ids = torch.tensor([pad_to_len(x, pad_id) for x in input_ids_list], dtype=torch.long)
        labels = torch.tensor([pad_to_len(x, self.label_pad_token_id) for x in labels_list], dtype=torch.long)

        attention_mask = (input_ids != pad_id).long()

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "index": torch.tensor(indices, dtype=torch.long),
        }
def parse_args():
    parser = argparse.ArgumentParser(description="Step 1: Compute Reference Loss (DDP)")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--dataset_field", type=str, nargs="+", default=["query", "response"])
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--bs", type=int, default=1)
    args = parser.parse_args()
    args.model = args.model_path 
    return args

def main():
    args = parse_args()

    rank, world_size = init_distributed()
    device = torch.device(f"cuda:{rank}") if torch.cuda.is_available() else "cpu"

    model, tokenizer = create_model_tokenizer_it(args)
    model = model.to(device, dtype=torch.bfloat16)
    model.eval()

    raw_dataset = load_and_preprocess_it(tokenizer=tokenizer, args=args)
    dataset = IndexedDataset(raw_dataset)
    sampler = DistributedSampler(dataset, shuffle=False) 
    collator = DataCollatorWithIndexPad(
    tokenizer=tokenizer,
    max_seq_length=args.max_seq_length,
    label_pad_token_id=IGNORE_INDEX,
    )

    dataloader = DataLoader(
    dataset,
    batch_size=args.bs,
    sampler=sampler,
    collate_fn=collator, 
    pin_memory=True,
    drop_last=False
    )


    criterion = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=IGNORE_INDEX)
    
    local_indices = []
    local_losses = []
    iterator = tqdm(dataloader, desc="Computing", disable=(rank != 0))
    
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            indices = batch.pop("index")
            local_indices.append(indices.cpu().numpy())
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device) 
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits

                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

                token_losses = criterion(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                ).view(shift_labels.size())

            padding_mask = shift_labels.eq(IGNORE_INDEX)
            num_active = torch.clamp((~padding_mask).sum(dim=-1), min=1)
            sample_losses = token_losses.sum(dim=-1) / num_active
            
            local_losses.append(sample_losses.float().cpu().numpy())

            del outputs, logits, shift_logits, token_losses, input_ids, labels
            if i % 100 == 0:
                torch.cuda.empty_cache()

    local_indices = np.concatenate(local_indices)
    local_losses = np.concatenate(local_losses)
    
    local_idx_tensor = torch.tensor(local_indices, device=device).long()
    local_loss_tensor = torch.tensor(local_losses, device=device).float()

    local_size = torch.tensor([local_idx_tensor.shape[0]], device=device)
    all_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(all_sizes, local_size)
    max_size = max([s.item() for s in all_sizes])
    pad_size = max_size - local_size.item()
    if pad_size > 0:
        idx_padding = torch.zeros(pad_size, device=device, dtype=torch.long)
        loss_padding = torch.zeros(pad_size, device=device, dtype=torch.float)
        local_idx_tensor = torch.cat([local_idx_tensor, idx_padding])
        local_loss_tensor = torch.cat([local_loss_tensor, loss_padding])
    gathered_indices = [torch.zeros_like(local_idx_tensor) for _ in range(world_size)]
    gathered_losses = [torch.zeros_like(local_loss_tensor) for _ in range(world_size)]
    
    dist.all_gather(gathered_indices, local_idx_tensor)
    dist.all_gather(gathered_losses, local_loss_tensor)
    if rank == 0:
        all_indices = torch.cat(gathered_indices).cpu().numpy()
        all_losses = torch.cat(gathered_losses).cpu().numpy()
        total_samples = len(raw_dataset)
        
        final_losses = np.zeros(total_samples, dtype=np.float32)
        
        unique_indices, unique_pos = np.unique(all_indices, return_index=True)
        valid_idx_mask = unique_indices < total_samples
        final_idx = unique_indices[valid_idx_mask]
        final_val = all_losses[unique_pos][valid_idx_mask]
        
        final_losses[final_idx] = final_val
        
        os.makedirs(args.output_dir, exist_ok=True)
        save_path = os.path.join(args.output_dir, "ref_losses.npy")
        np.save(save_path, final_losses)

if __name__ == "__main__":
    main()