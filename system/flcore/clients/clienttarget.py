import numpy as np
import time
import os
import torch
import glob
from PIL import Image
from flcore.clients.clientbase import Client
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# --- Custom Dataset to load generated images ---
class SyntheticDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # Find all png files
        self.image_paths = glob.glob(os.path.join(root_dir, "*.png"))
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        # Return only image (TARGET typically treats synthetic data as unlabeled for KD)
        return image 

class clientTARGET(Client):
    def __init__(self, args, id, train_data, **kwargs):
        super().__init__(args, id, train_data, **kwargs)

        self.nums = 8000
        self.syn_data_loader = None
        self.old_network = None
        self.kd_alpha = 25
        self.synthtic_save_dir = "dataset/synthetic_data"
        
        # Define transforms for synthetic data (must match training transforms)
        # Note: If generator outputs [-1,1] and we saved it normalized, 
        # we might need to normalize again here.
        self.syn_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ])

    def train(self, task=None):
        trainloader = self.load_train_data(task=task)
        self.model.train()
        start_time = time.time()

        # Create iterator for synthetic data if available
        syn_iter = None
        if self.syn_data_loader is not None:
            syn_iter = iter(self.syn_data_loader)

        for epoch in range(self.local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]): x[0] = x[0].to(self.device)
                else: x = x.to(self.device)
                y = y.to(self.device)
                
                output = self.model(x)
                loss = self.loss(output, y)
                
                # --- KD with Synthetic Data ---
                if self.syn_data_loader is not None and self.old_network is not None:
                    try:
                        syn_inputs = next(syn_iter)
                    except StopIteration:
                        syn_iter = iter(self.syn_data_loader)
                        syn_inputs = next(syn_iter)
                    
                    syn_inputs = syn_inputs.to(self.device)
                    
                    syn_outputs = self.model(syn_inputs)
                    with torch.no_grad():
                        syn_old_outputs = self.old_network(syn_inputs)
                    
                    # Distill the RESPONSE of the old network
                    kd_loss = self.KD_loss_func(syn_outputs, syn_old_outputs, 2)
                    loss += self.kd_alpha * kd_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def KD_loss_func(self, logits, targets, T=2.0):
        """Standard KL Div for Distillation"""
        import torch.nn.functional as F
        q = F.log_softmax(logits / T, dim=1)
        p = F.softmax(targets / T, dim=1)
        return F.kl_div(q, p, reduction='batchmean') * (T * T)
            
    def get_syn_data_loader(self):
        # We load data from the PREVIOUS task (task - 1)
        # Because we want to remember what we learned before.
        target_task_id = self.current_task - 1
        if target_task_id < 0: return None

        data_dir = os.path.join(self.synthtic_save_dir, f"task_{target_task_id}")
        
        if not os.path.exists(data_dir):
            print(f"[Client {self.id}] Warning: No synthetic data found at {data_dir}")
            return None

        # Calculate batch size for synthetic data
        # Heuristic: roughly same number of batches as real data
        syn_bs = self.args.batch_size 

        syn_dataset = SyntheticDataset(data_dir, transform=self.syn_transform)
        
        syn_data_loader = DataLoader(
            syn_dataset, batch_size=syn_bs, shuffle=True,
            num_workers=2, drop_last=True
        )
        print(f"[Client {self.id}] Loaded {len(syn_dataset)} synthetic images from {data_dir}")
        return syn_data_loader