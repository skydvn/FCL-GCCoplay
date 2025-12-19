import time
import torch
import torch.nn.functional as F
import copy
import numpy as np
from torch import nn, optim
from flcore.servers.serverbase import Server
from flcore.clients.ours_v2 import clientOursV2

# ==========================================
# 1. GENERATOR CLASS 
# ==========================================
class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, img_size=32, nc=3, num_classes=100, device=None):
        super(Generator, self).__init__()
        self.params = (nz, ngf, img_size, nc, num_classes) 
        self.init_size = img_size // 4
        self.device = device
        self.label_emb = nn.Embedding(num_classes, nz)
        self.l1 = nn.Sequential(nn.Linear(nz * 2, ngf * 2 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(ngf * 2),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ngf*2, ngf*2, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ngf*2, ngf, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf, nc, 3, stride=1, padding=1),
            nn.Sigmoid(),  
        )

    def forward(self, z, labels):
        c = self.label_emb(labels)
        gen_input = torch.cat((z, c), -1)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

    def clone(self):
        clone = Generator(self.params[0], self.params[1], self.params[2], self.params[3], self.params[4], device=self.device)
        clone.load_state_dict(self.state_dict())
        return clone.to(self.device)

def weight_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None: nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


# ==========================================
# 2. SERVER CLASS
# ==========================================
class OursV2(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.Budget = []
        
        # --- Config Generator ---
        if 'cifar' in self.dataset.lower():
            self.img_size = 32; self.nz = 256
        elif 'imagenet' in self.dataset.lower():
            self.img_size = 64 if '100' in self.dataset else 224; self.nz = 256
        else:
            self.img_size = 64; self.nz = 100

        self.global_generator = Generator(
            nz=self.nz, ngf=64, img_size=self.img_size, nc=3, device=self.device, 
            num_classes=args.num_classes,
        ).to(self.device)

        # [NEW] Placeholder for the Anchor (Previous Generator)
        self.prev_generator = None
        self.has_initialized_generator = False 
        
        # Hyperparameters
        self.g_lr = getattr(args, 'g_lr', 0.001)     
        self.c_lr = getattr(args, 'c_lr', 0.001)     
        self.g_steps = getattr(args, 'g_steps', 100) 
        self.k_steps = getattr(args, 'k_steps', 100) 
        self.batch_size_gen = 64
        self.T = getattr(args, 'T', 2.0)             
        
        self.set_clients(clientOursV2)
        print(f"Server Initialized. Generator Config: nz={self.nz}")

    def train(self):
        for task in range(self.args.num_tasks):
            print(f"\n================ Current Task: {task} =================")
            self.current_task = task
            self._update_label_info()
            if task > 0: self._load_task_data_for_clients(task)

            # [ANCHOR SETUP] Before starting a new task (if > 0), snapshot the generator
            if task > 0 and self.prev_generator is None:
                print(f">>> [Task {task}] Creating Anchor from Previous Task Generator...")
                self.prev_generator = self.global_generator.clone()
                self.prev_generator.eval()
                for p in self.prev_generator.parameters(): p.requires_grad = False

            # -----------------------------------------------
            # Loop Rounds
            # -----------------------------------------------
            for i in range(self.global_rounds):
                s_t = time.time()
                glob_iter = i + self.global_rounds * task
                
                # 1. Send Models
                self.selected_clients = self.select_clients()
                self.send_models() 

                if i % self.eval_gap == 0:
                    print(f"\n-------------Round number: {i}-------------")
                    self.eval(task=task, glob_iter=glob_iter, flag="global")

                # 2. Local Training
                for client in self.selected_clients:
                    client.train(task=task)

                if i % self.eval_gap == 0:
                    self.eval(task=task, glob_iter=glob_iter, flag="local")

                # 3. Receive Models
                self.receive_models() 
                
                # -----------------------------------------------
                # 4. Aggregation Logic (Task 0 vs Task > 0)
                # -----------------------------------------------
                
                # Step A: Always do FedAvg (Base Aggregation)
                self.aggregate_parameters()

                # Step B: If Task > 0, we ADD Distillation during the rounds
                # "I want to use distillation... combine this with averaging"
                if task > 0 and i < self.global_rounds - 1:
                    # Run a lighter version of KD (e.g., 50 steps) to mix past knowledge
                    # into the current FedAvg weights
                    self.train_global_classifier(self.available_labels_current, steps=50)

                # -----------------------------------------------
                # 5. End of Task Logic (Round R)
                # -----------------------------------------------
                if i == self.global_rounds - 1:
                    print(f"\n>>> [Round R] End of Task {task} detected.")
                    
                    # A. Train Generator
                    # Task 0: Train normally. 
                    # Task > 0: Train with Anchor.
                    use_anchor = (task > 0)
                    self.train_global_generator(self.available_labels_current, use_anchor=use_anchor)

                    # B. Train Classifier (Full Distillation)
                    # Now that Generator is updated, distill knowledge into Classifier
                    self.train_global_classifier(self.available_labels_current, steps=self.k_steps)

                    # C. Update Anchor for NEXT task
                    # We take the freshly trained generator and save it as the new anchor
                    self.prev_generator = self.global_generator.clone()
                    self.prev_generator.eval()
                    for p in self.prev_generator.parameters(): p.requires_grad = False
                
                self.Budget.append(time.time() - s_t)
                print(f"Round {i} Time: {self.Budget[-1]:.2f}s")

            # Final Eval
            if self.args.offlog:
                print(f"\n>>> Task {task} Finished. Evaluating Forgetting Rate...")
                # Gửi model mới nhất xuống client để test
                self.send_models() 
                # Hàm eval_task trong ServerBase sẽ tính toán và log Forgetting
                self.eval_task(task=task, glob_iter=glob_iter, flag="global")

            self.change_task(task, (task + 1)*self.global_rounds)


    def train_global_generator(self, available_labels, use_anchor=False):
        labels_list = list(available_labels)
        if len(labels_list) == 0: return

        if not self.has_initialized_generator:
            self.global_generator.apply(weight_init)
            self.has_initialized_generator = True
        
        self.global_generator.train()
        
        # Freeze Teachers
        teachers = self.uploaded_models
        for t in teachers: t.eval(); 
        for p in t.parameters(): p.requires_grad = False

        optimizer_g = optim.Adam(self.global_generator.parameters(), lr=self.g_lr)
        criterion_gan = nn.CrossEntropyLoss()
        mse_loss = nn.MSELoss() # [NEW] For Anchor
        
        # Anchor Strength
        alpha = 10.0 

        print(f"   -> Training Generator (Anchor={use_anchor})...")
        for step in range(self.g_steps):
            optimizer_g.zero_grad()
            
            # Sample
            labels = np.random.choice(labels_list, self.batch_size_gen)
            labels = torch.tensor(labels).long().to(self.device)
            z = torch.randn(self.batch_size_gen, self.nz).to(self.device)

            gen_imgs = self.global_generator(z, labels)

            # 1. Classification Loss (Teachers)
            total_loss = 0
            for teacher in teachers:
                preds = teacher(gen_imgs)
                total_loss += criterion_gan(preds, labels)
            loss_g = total_loss / len(teachers)

            # 2. [NEW] Anchor Loss
            if use_anchor and self.prev_generator is not None:
                with torch.no_grad():
                    # Generate images using the OLD generator with SAME noise
                    anchor_imgs = self.prev_generator(z, labels)
                
                # MSE ensures structural similarity (prevents forgetting old classes)
                loss_anchor = mse_loss(gen_imgs, anchor_imgs)
                loss_g += alpha * loss_anchor

            loss_g.backward()
            optimizer_g.step()

    def train_global_classifier(self, available_labels, steps=100):
        labels_list = list(available_labels)
        if len(labels_list) == 0: return

        self.global_model.train() 
        for param in self.global_model.parameters(): param.requires_grad = True
        self.global_generator.eval()

        teachers = self.uploaded_models
        for t in teachers: t.eval(); 
        for p in t.parameters(): p.requires_grad = False
            
        optimizer_c = optim.Adam(self.global_model.parameters(), lr=self.c_lr)
        criterion_kd = nn.KLDivLoss(reduction='batchmean')

        # print(f"   -> Distilling into Classifier ({steps} steps)...")
        for step in range(steps):
            labels = np.random.choice(labels_list, self.batch_size_gen)
            labels = torch.tensor(labels).long().to(self.device)
            z = torch.randn(self.batch_size_gen, self.nz).to(self.device)

            with torch.no_grad():
                gen_imgs = self.global_generator(z, labels)

            # Teacher Ensemble
            teacher_logits_sum = 0
            with torch.no_grad():
                for teacher in teachers:
                    teacher_logits_sum += teacher(gen_imgs)
            teacher_avg_logits = teacher_logits_sum / len(teachers)

            # Student
            student_logits = self.global_model(gen_imgs)

            # KL Loss
            loss_kd = criterion_kd(
                F.log_softmax(student_logits / self.T, dim=1),
                F.softmax(teacher_avg_logits / self.T, dim=1)
            ) * (self.T * self.T)

            optimizer_c.zero_grad()
            loss_kd.backward()
            optimizer_c.step()

    def send_models(self):
        for client in self.clients:
            client.set_parameters(self.global_model)
            client.set_generator_parameters(self.global_generator)

    def _update_label_info(self):
        available_labels = set()
        for u in self.clients:
            available_labels = available_labels.union(set(u.classes_so_far))
        self.available_labels_current = available_labels

    def _load_task_data_for_clients(self, task):
        # (Assuming your imports are correct here)
        if self.args.dataset == 'IMAGENET1k':
            from utils.data_utils import read_client_data_FCL_imagenet1k as read_func
        elif 'CIFAR100' in self.args.dataset:
            from utils.data_utils import read_client_data_FCL_cifar100 as read_func
        else:
            from utils.data_utils import read_client_data_FCL_cifar10 as read_func
        
        for i, client in enumerate(self.clients):
            train_data, label_info = read_func(
                i, task=task, classes_per_task=self.args.cpt, count_labels=True
            )
            client.next_task(train_data, label_info)