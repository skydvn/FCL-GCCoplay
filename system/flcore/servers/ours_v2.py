import time
import torch
import torch.nn.functional as F
import copy
import numpy as np
from torch import nn, optim
from flcore.servers.serverbase import Server
from flcore.clients.ours_v2 import clientOursV2
from flcore.utils_core.target_utils import *
from utils.data_utils import read_client_data_FCL_cifar100, read_client_data_FCL_imagenet1k, read_client_data_FCL_cifar10

def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None: nn.init.constant_(m.bias, 0)

# ==========================================
# 2. SERVER CLASS (Finalized)
# ==========================================
class OursV2(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.Budget = []
        
        # --- Config Generator ---
        # Tự động chỉnh size dựa trên dataset
        if 'cifar100' in self.dataset.lower():
            self.img_size = 32
            self.nz = 512 # Tăng nz lên chút cho Cifar100
        elif 'imagenet' in self.dataset.lower():
            self.img_size = 64 if '100' in self.dataset else 224 # ImageNet-R thường resize về 64 hoặc 224
            self.nz = 256
        else:
            self.img_size = 32
            self.nz = 100

        self.global_generator = Generator(
            nz=self.nz, ngf=64, img_size=self.img_size, nc=3, 
            num_classes=args.num_classes, device=self.device
        ).to(self.device)
        
        # Khởi tạo weight ngay từ đầu
        self.global_generator.apply(weight_init)

        # [NEW] Placeholder for the Anchor (Previous Generator)
        self.prev_generator = None
        
        # Hyperparameters
        self.g_lr = getattr(args, 'g_lr', 0.0002) # GAN thường dùng LR thấp hơn (0.0002 là chuẩn của Adam)
        self.c_lr = getattr(args, 'c_lr', 0.001)     
        self.g_steps = getattr(args, 'g_steps', 100) 
        self.k_steps = getattr(args, 'k_steps', 200) # Tăng step distillation lên chút
        self.batch_size_gen = 64
        self.T = getattr(args, 'T', 2.0)        

        self.optimizer_g = optim.Adam(self.global_generator.parameters(), lr=self.g_lr, betas=(0.5, 0.999))
        self.optimizer_c = optim.Adam(self.global_model.parameters(), lr=self.c_lr)

        self.scheduler_g = optim.lr_scheduler.MultiStepLR(self.optimizer_g, milestones=[30, 80], gamma=0.1)
        self.scheduler_c = optim.lr_scheduler.MultiStepLR(self.optimizer_c, milestones=[30, 80], gamma=0.1)


        self.set_clients(clientOursV2)
        print(f"Server Initialized. Generator Config: ImgSize={self.img_size}, nz={self.nz}, Device={self.device}")

    def train(self):
        for task in range(self.args.num_tasks):

            print(f"\n================ Current Task: {task} =================")
            if task == 0:
                 # update labels info. for the first task
                available_labels = set()
                available_labels_current = set()
                available_labels_past = set()
                for u in self.clients:
                    available_labels = available_labels.union(set(u.classes_so_far))
                    available_labels_current = available_labels_current.union(set(u.current_labels))
                # print("ahihi " + str(len(available_labels_current)))
                for u in self.clients:
                    u.available_labels = list(available_labels)
                    u.available_labels_current = list(available_labels_current)
                    u.available_labels_past = list(available_labels_past)

            else:
                self.current_task = task
                
                torch.cuda.empty_cache()
                for i in range(len(self.clients)):
                    
                    if self.args.dataset == 'IMAGENET1k':
                        train_data, label_info = read_client_data_FCL_imagenet1k(i, task=task, classes_per_task=self.args.cpt, count_labels=True)
                    elif self.args.dataset == 'CIFAR100':
                        train_data, label_info = read_client_data_FCL_cifar100(i, task=task, classes_per_task=self.args.cpt, count_labels=True)
                    elif self.args.dataset == 'CIFAR10':
                        train_data, label_info = read_client_data_FCL_cifar10(i, task=task, classes_per_task=self.args.cpt, count_labels=True)
                    else:
                        raise NotImplementedError("Not supported dataset")

                    # update dataset
                    self.clients[i].next_task(train_data, label_info) # assign dataloader for new data
                    # print(self.clients[i].task_dict)

                # update labels info.
                available_labels = set()
                available_labels_current = set()
                available_labels_past = self.clients[0].available_labels
                for u in self.clients:
                    available_labels = available_labels.union(set(u.classes_so_far))
                    available_labels_current = available_labels_current.union(set(u.current_labels))

                for u in self.clients:
                    u.available_labels = list(available_labels)
                    u.available_labels_current = list(available_labels_current)
                    u.available_labels_past = list(available_labels_past)

            for i in range(self.global_rounds):
                print(f"\n-------------Round number: {i}-------------")
                s_t = time.time()
                glob_iter = i + self.global_rounds * task
                
                # 1. Send Models
                self.selected_clients = self.select_clients() 
                
                self.eval(task=task, glob_iter=glob_iter, flag="global")

                # 2. Local Training
                for client in self.selected_clients:
                    client.train(task=task)
                if i % self.eval_gap == 0:
                    self.eval(task=task, glob_iter=glob_iter, flag="local")
                self.receive_models()

                self.train_global_generator() # Updates Generator
                self.aggregate_parameters()
                self.eval(task=task, glob_iter=glob_iter, flag="local")
                self.Budget.append(time.time() - s_t)
                print(f"Round {i} Time: {self.Budget[-1]:.2f}s")
            self.receive_models()
            self.train_global_generator()
            self.train_global_classifier()
            print(f"\n>>> {task} task(s) is/are finished. Evaluating Forgetting Rate...")
            self.send_models() 
            self.eval(task=task, glob_iter=glob_iter, flag="global")
            self.eval_task(task=task, glob_iter=task, flag="global")

            # self.change_task(task, (task + 1)*self.global_rounds)


    

    def train_global_generator(self):
        print(f"[Server-side] Start training Generator")
        
        # --- 1. Label & Model Preparation ---
        # We process client labels once here to ensure they are standard Numpy arrays 
        # (Fixes the mismatch between Tensor/List/Numpy that causes 0 loss)
        processed_client_labels = {}
        all_labels_set = set()

        for client_id, info in self.client_info_dict.items():
            raw_labels = info["label"]
            
            # 1. Handle Torch Tensors
            if isinstance(raw_labels, torch.Tensor):
                clean_labels = raw_labels.cpu().numpy().astype(int)
            
            # 2. Handle Lists (which might contain Tensors)
            elif isinstance(raw_labels, list):
                if len(raw_labels) > 0 and isinstance(raw_labels[0], torch.Tensor):
                    clean_labels = np.array([x.item() for x in raw_labels], dtype=int)
                else:
                    clean_labels = np.array(raw_labels, dtype=int)
            
            # 3. FIX: Handle Sets (This is where your error happened)
            elif isinstance(raw_labels, set):
                clean_labels = np.array(list(raw_labels), dtype=int)
                
            # 4. Fallback for other iterables
            else:
                clean_labels = np.array(list(raw_labels), dtype=int)
            
            processed_client_labels[client_id] = clean_labels
            all_labels_set.update(clean_labels.tolist())

            # Optimization: Set teacher models to eval/frozen ONCE outside the loop
            info["model"].eval()
            for param in info["model"].parameters():
                param.requires_grad = False

        labels_list = list(all_labels_set)
        if len(labels_list) == 0:
            print("[Server-side] No labels found. Skipping generator training.")
            return

        print(f"[Server-side] Generator training on {len(labels_list)} unique labels with {labels_list}.")

        # --- 2. Generator Setup ---
        self.global_generator.train()
        criterion_ce = nn.CrossEntropyLoss()
        mse_loss = nn.MSELoss() 
        alpha = 10.0 

        # --- 3. Training Loop ---
        for step in range(self.g_steps):
            self.optimizer_g.zero_grad()
            
            # A. Sample Noise and Labels
            selected_labels = np.random.choice(labels_list, self.batch_size_gen)
            labels_tensor = torch.tensor(selected_labels, dtype=torch.long).to(self.device)
            z = torch.randn(self.batch_size_gen, self.nz).to(self.device)

            # B. Generate Images
            gen_imgs = self.global_generator(z, labels_tensor)

            # C. Classification Loss (Teacher Feedback)
            total_ce_loss = torch.tensor(0.0, device=self.device)
            valid_teachers_count = 0
            
            for client_id, info in self.client_info_dict.items():
                teacher_model = info["model"]
                teacher_labels = processed_client_labels[client_id] # Use the pre-cleaned numpy array

                # Boolean mask: True if the selected label exists in this teacher's dataset
                mask = np.isin(selected_labels, teacher_labels)
                
                if mask.sum() > 0:
                    valid_teachers_count += 1
                    mask_tensor = torch.tensor(mask, device=self.device)
                    
                    # Filter: Only pass images corresponding to labels this teacher knows
                    relevant_imgs = gen_imgs[mask_tensor]
                    relevant_labels = labels_tensor[mask_tensor]
                    
                    # Teacher Prediction
                    preds = teacher_model(relevant_imgs)
                    
                    # Accumulate Loss
                    total_ce_loss += criterion_ce(preds, relevant_labels)

            # D. Combine Losses
            if valid_teachers_count > 0:
                loss_g = total_ce_loss
            else:
                # Fallback: maintain graph connectivity but 0 value
                loss_g = torch.tensor(0.0, device=self.device, requires_grad=True)

            # E. Anchor Loss (Regularization with previous generator)
            if self.prev_generator is not None:
                self.prev_generator.eval()
                # Ensure prev_generator is frozen (safety check)
                for param in self.prev_generator.parameters():
                    param.requires_grad = False
                    
                with torch.no_grad():
                    anchor_imgs = self.prev_generator(z, labels_tensor)
                
                loss_anchor = mse_loss(gen_imgs, anchor_imgs)
                loss_g = loss_g + alpha * loss_anchor 

            # F. Backpropagation
            loss_g.backward()
            self.optimizer_g.step()
            self.scheduler_g.step()
            # Optional: Print periodically instead of every step
            if step % 10 == 0:
                print(f"[Server-side] Generator update: Step {step}: Loss={loss_g.item():.4f} | Teachers Used={valid_teachers_count} | Lr: {self.optimizer_g.param_groups[0]['lr']}")
        
        # --- 4. Update Anchor ---
        self.prev_generator = copy.deepcopy(self.global_generator)

    def train_global_classifier(self, steps=100):
        print(f"[Server-side] Start training Global Classifier")
        available_labels = []
        for _, v in self.client_info_dict.items():
            label = list(v["label"])
            available_labels.extend(label)
        labels_list = list(set(available_labels))
        # 1. Setup Student (Global Model)
        self.global_model.train() 
        for param in self.global_model.parameters(): 
            param.requires_grad = True
        
        # 2. Setup Generator (Fixed)
        self.global_generator.eval()
        for step in range(steps):
            self.optimizer_c.zero_grad()

            # --- A. Generate Synthetic Data ---
            # Randomly sample labels from the pool of all available labels
            selected_labels = np.random.choice(labels_list, self.batch_size_gen)
            labels_tensor = torch.tensor(selected_labels).long().to(self.device)
            z = torch.randn(self.batch_size_gen, self.nz).to(self.device)

            # with torch.no_grad():
            gen_imgs = self.global_generator(z, labels_tensor)            

            # --- B. Student Forward Pass ---
            student_logits = self.global_model(gen_imgs)

            # --- C. Teacher Ensemble (Selective Distillation) ---
            teacher_logits_sum = torch.zeros_like(student_logits)
            # Count how many teachers contributed to each image in the batch
            # Shape: [Batch_Size, 1]
            teacher_counts = torch.zeros(self.batch_size_gen, 1, device=self.device)
            
            # Iterate over all available client models (teachers)
            for client_id, info in self.client_info_dict.items():
                teacher_model = info["model"]
                teacher_known_labels = info["label"] # List of labels this client knows

                # Create a Boolean Mask: [Batch_Size]
                # True if the image's label is known by this client
                # We use numpy.isin for efficiency on the label array
                mask_np = np.isin(selected_labels, teacher_known_labels)
                
                # If this client knows at least one label in the batch
                if mask_np.sum() > 0:
                    mask_tensor = torch.tensor(mask_np, device=self.device).unsqueeze(1) # Shape [B, 1]

                    teacher_model.eval()
                    # with torch.no_grad():
                        # Get logits for the whole batch (computationally cheaper than slicing usually)
                    logits = teacher_model(gen_imgs)
                    
                    # Accumulate logits ONLY where mask is True
                    # If mask is False (0), we add 0.0
                    teacher_logits_sum += logits * mask_tensor
                    teacher_counts += mask_tensor

            # --- D. Compute Loss ---
            # Avoid division by zero (if a label somehow has 0 teachers, though unlikely if logic is correct)
            teacher_counts[teacher_counts == 0] = 1.0 
            
            # Average the logits to get the "Ensemble Teacher"
            teacher_avg_logits = teacher_logits_sum / teacher_counts

            # KL Divergence Loss
            # We assume the "Teacher" is the Softmax of the averaged logits
            loss_kd = KD_loss(student_logits, teacher_avg_logits, self.T)

            loss_kd.backward()
            self.optimizer_c.step()
            self.scheduler_c.step()
            if step % 10 == 0:
                print(f"[Server-side] Generator update: Step {step}: Loss={loss_kd.item():.4f} | Lr: {self.optimizer_c.param_groups[0]['lr']}")
    def send_models(self):
        for client in self.clients:
            client.set_parameters(self.global_model)
            # Nếu client cần dùng Generator để Replay cục bộ (nếu thuật toán yêu cầu)
            client.set_generator_parameters(self.global_generator)

    def receive_models(self):
        print(f"[Server-side] Receive models and lists of labels from clients")
        self.client_info_dict = {}
        for client in self.clients:
            self.client_info_dict[client.id] = {
                "model": client.model,
                "label": client.unique_labels
            }
            self.uploaded_models.append(client.model)

    # def aggregate_parameters(self):
    #     uploaded_models = []
    #     for client_id in self.client_info_dict:
    #         model = self.client_info_dict[client_id]["model"]
    #         uploaded_models.append(model)

        