import time
import torch
import copy
import os
import shutil
import numpy as np
from torch import nn, optim
from torchvision.utils import save_image
from flcore.clients.clienttarget import clientTARGET
from flcore.servers.serverbase import Server
# Ensure this import points to your new Conditional Generator file
# from models.generator import Generator 
from flcore.utils_core.target_utils import *
from utils.data_utils import *
from copy import deepcopy

# --- Helper Functions for Generator Training ---
class BNFeatureHook:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.r_feature = None 

    def hook_fn(self, module, input, output):
        self.r_feature = input[0]

    def remove(self):
        self.hook.remove()

def get_bn_loss(teacher_model, gen_imgs):
    """Calculates BN statistics distance between real (stored in teacher) and fake data."""
    bn_hooks = []
    bn_layers = [m for m in teacher_model.modules() if isinstance(m, nn.BatchNorm2d)]

    for module in bn_layers:
        bn_hooks.append(BNFeatureHook(module))

    teacher_model(gen_imgs)

    loss_bn = 0.0
    for hook, layer in zip(bn_hooks, bn_layers):
        real_mean = layer.running_mean
        real_var = layer.running_var
        gen_feat = hook.r_feature
        gen_mean = torch.mean(gen_feat, dim=[0, 2, 3])
        gen_var = torch.var(gen_feat, dim=[0, 2, 3], unbiased=False)
        loss_bn += torch.norm(gen_mean - real_mean, 2) + torch.norm(gen_var - real_var, 2)

    for hook in bn_hooks: hook.remove()
    return loss_bn

# ------------------------------------------------

class FedTARGET(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.synthtic_save_dir = "dataset/synthetic_data"
        if os.path.exists(self.synthtic_save_dir):
            shutil.rmtree(self.synthtic_save_dir)
        
        self.nums = 8000 # Number of images to generate per task
        self.available_labels_current = []
        
        # --- Config Generator ---
        # Heuristic for image size
        if 'cifar' in self.dataset.lower():
            self.img_size = 32; self.nz = 256
        elif 'imagenet' in self.dataset.lower():
            self.img_size = 64 if '100' in self.dataset else 224; self.nz = 256
        else:
            self.img_size = 32; self.nz = 100

        # Initialize CONDITIONAL Generator
        self.global_generator = Generator(
            nz=self.nz, ngf=64, img_size=self.img_size, nc=3, 
            num_classes=args.num_classes, device=self.device
        ).to(self.device)

        self.set_clients(clientTARGET)
        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")
        self.Budget = []

    def train(self):
        for task in range(self.args.num_tasks):
            print(f"\n================ Current Task: {task} =================")
            
            # --- 1. Label Collection ---
            if task == 0:
                available_labels = set()
                available_labels_current = set()
                for u in self.clients:
                    available_labels.update(u.classes_so_far)
                    available_labels_current.update(u.current_labels)
                
                self.available_labels_current = list(available_labels_current)
                for u in self.clients:
                    u.available_labels = list(available_labels)
                    u.available_labels_current = list(available_labels_current)
            else:
                self.current_task = task
                torch.cuda.empty_cache()

                # Freeze old network for distillation
                self.old_network = deepcopy(self.global_model)
                for p in self.old_network.parameters(): p.requires_grad = False

                # Update Clients
                for i in range(len(self.clients)):
                    if self.args.dataset == 'IMAGENET1k':
                        read_func = read_client_data_FCL_imagenet1k
                    elif 'cifar100' in self.args.dataset.lower():
                        read_func = read_client_data_FCL_cifar100
                    else:
                        raise NotImplementedError("Dataset not supported")

                    train_data, label_info = read_func(i, task=task, classes_per_task=self.args.cpt, count_labels=True)
                    self.clients[i].next_task(train_data, label_info)
                    
                    # Store client's old model
                    self.clients[i].old_network = deepcopy(self.clients[i].model)
                    for p in self.clients[i].old_network.parameters(): p.requires_grad = False

                # Recalculate Labels
                available_labels = set()
                available_labels_current = set()
                for u in self.clients:
                    available_labels.update(u.classes_so_far)
                    available_labels_current.update(u.current_labels)
                
                self.available_labels_current = list(available_labels_current)
                for u in self.clients:
                    u.available_labels = list(available_labels)
                    u.available_labels_current = list(available_labels_current)

            # --- 2. Training Loop ---
            for i in range(self.global_rounds):
                # Prepare clients for synthetic data usage (from previous task)
                if task > 0:
                    for u in self.clients:
                        u.syn_data_loader = u.get_syn_data_loader()
                        # u.it = iter(u.syn_data_loader) # Often safer to use iter() directly in loop

                glob_iter = i + self.global_rounds * task
                s_t = time.time()
                
                self.selected_clients = self.select_clients()
                self.send_models()
                self.eval(task=task, glob_iter=glob_iter, flag="global")

                for client in self.selected_clients:
                    client.train(task=task)

                self.receive_models()
                self.aggregate_parameters()

                self.Budget.append(time.time() - s_t)
                print(f"Round {i} Time: {self.Budget[-1]:.2f}")

            # --- 3. Data Generation (Target Approach) ---
            # Train generator to capture knowledge of CURRENT task
            print(f">>> Task {task} Finished. Generating Synthetic Data...")
            self.data_generation(task=task, available_labels=self.available_labels_current)

            if int(task/self.N_TASKS) == int(self.args.num_tasks/self.N_TASKS-1):
                self.eval_task(task=task, glob_iter=glob_iter, flag="global")

    def data_generation(self, task, available_labels):
        """
        Trains the Conditional Generator and saves images to disk.
        """
        save_dir = os.path.join(self.synthtic_save_dir, f"task_{task}")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 1. Setup Training
        self.global_model.eval()
        for p in self.global_model.parameters(): p.requires_grad = False
        
        self.global_generator.train()
        # Reset generator weights for new task (optional, but often helps stability)
        # self.global_generator.apply(weight_init) 
        
        optimizer_g = optim.Adam(self.global_generator.parameters(), lr=0.001, betas=(0.5, 0.999))
        
        # Hyperparams
        g_steps = 200   # How many steps to train the generator
        batch_size = 64
        beta_bn = 1.0   # Weight for BN statistics loss
        
        print(f"[Server] Training Generator on {len(available_labels)} classes...")

        # 2. Train Generator
        for step in range(g_steps):
            optimizer_g.zero_grad()

            # Sample Labels from current task
            targets = np.random.choice(available_labels, batch_size)
            targets = torch.tensor(targets, dtype=torch.long, device=self.device)
            
            # Sample Noise
            z = torch.randn(batch_size, self.nz, device=self.device)
            
            # Generate
            gen_imgs = self.global_generator(z, targets)
            
            # Loss 1: Class Confidence (Maximize confidence of Teacher on generated images)
            outputs = self.global_model(gen_imgs)
            loss_ce = nn.CrossEntropyLoss()(outputs, targets)
            
            # Loss 2: BN Statistics Matching
            loss_bn = get_bn_loss(self.global_model, gen_imgs)

            loss = loss_ce + beta_bn * loss_bn
            loss.backward()
            optimizer_g.step()

            if step % 50 == 0:
                print(f"Iter {step}: Loss={loss} (CE={loss_ce}, BN={loss_bn})")

        # 3. Save Synthetic Images
        print(f"[Server] Saving {self.nums} synthetic images...")
        self.global_generator.eval()
        
        count = 0
        with torch.no_grad():
            while count < self.nums:
                # Generate in batches
                curr_bs = min(batch_size, self.nums - count)
                
                # Balanced generation across available labels
                targets = np.random.choice(available_labels, curr_bs)
                targets_tensor = torch.tensor(targets, dtype=torch.long, device=self.device)
                z = torch.randn(curr_bs, self.nz, device=self.device)
                
                gen_imgs = self.global_generator(z, targets_tensor)
                
                # Denormalize if your generator outputs [-1, 1] but you save as png
                # Assuming standard save_image handles [0,1], we map [-1, 1] -> [0, 1]
                gen_imgs = (gen_imgs + 1.0) / 2.0 

                for k in range(curr_bs):
                    # Save as: dataset/synthetic_data/task_0/img_0.png
                    save_path = os.path.join(save_dir, f"img_{count}.png")
                    save_image(gen_imgs[k], save_path)
                    count += 1