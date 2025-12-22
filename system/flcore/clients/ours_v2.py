import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import copy
from flcore.clients.clientbase import Client
from flcore.utils_core.target_utils import *
import torch
import torch.nn as nn

class clientOursV2(Client):
    def __init__(self, args, id, train_data, **kwargs):
        super().__init__(args, id, train_data, **kwargs)
        # Cấu hình Generator
        if 'cifar' in args.dataset.lower():
            self.img_size = 32; self.nz = 512
        elif 'imagenet' in args.dataset.lower():
            self.img_size = 64 if '100' in args.dataset else 224; self.nz = 256
        else:
            self.img_size = 64; self.nz = 100
            
        self.nc = 3
        # Trọng số cho Loss Replay và nhiệt độ T cho KD
        self.replay_weight = args.replay_weight if hasattr(args, 'replay_weight') else 1.0
        self.T = args.T if hasattr(args, 'T') else 2.0 

        # Khởi tạo Generator
        self.generator = Generator(
            nz=self.nz, ngf=64, img_size=self.img_size, nc=self.nc, device=self.device,
            num_classes=args.num_classes,
        ).to(self.device)
        self.freeze_generator()

        self.old_network = None # Teacher model

    def freeze_generator(self):
        self.generator.eval()
        for param in self.generator.parameters():
            param.requires_grad = False

    def set_generator_parameters(self, global_generator):
        self.generator.load_state_dict(global_generator.state_dict())
        self.freeze_generator()

    def next_task(self, train, label_info=None, if_label=True):
        """Override để lưu lại model cũ làm Teacher trước khi học task mới"""
        super().next_task(train, label_info, if_label)
        
        # Snapshot model hiện tại thành old_network (Teacher)
        self.old_network = copy.deepcopy(self.model)
        self.old_network.eval()
        for param in self.old_network.parameters():
            param.requires_grad = False

    def train(self, task=None):
        trainloader = self.load_train_data(task=task)
        for param in self.model.parameters():
            param.requires_grad = True
        self.model.train()
        self.generator.eval()
        
        start_time = time.time()
        local_loss = []
        for epoch in range(self.local_epochs):
            for i, (x_real, y_real) in enumerate(trainloader):
                self.optimizer.zero_grad()
                if isinstance(x_real, list): x_real = x_real[0]
                x_real = x_real.to(self.device)
                y_real = y_real.to(self.device)

                # --- 1. Real Data: CrossEntropy Loss ---
                output_real = self.model(x_real)
                loss_real = self.loss(output_real, y_real)

                # --- 2. Replay Data: KL Divergence Loss (Distillation) ---
                loss_replay = 0
                # Chỉ Replay khi có Teacher (old_network) và task > 0
                if self.old_network is not None:
                    batch_size = x_real.shape[0]
                    
                    # a. Sinh dữ liệu giả
                    z = torch.randn(batch_size, self.nz).to(self.device)
                    # Chọn nhãn từ các task CŨ để ôn tập
                    # Lưu ý: Teacher chỉ biết kiến thức cũ
                    if len(self.classes_past_task) > 0:
                        fake_labels = np.random.choice(self.classes_past_task, batch_size)
                        fake_labels = torch.tensor(fake_labels).long().to(self.device)
                        
                        with torch.no_grad():
                            syn_inputs = self.generator(z, fake_labels)
                            # Teacher (Old Model) dự đoán soft labels
                            with torch.no_grad():
                                syn_old_outputs = self.old_network(syn_inputs)

                        # Student (Current Model) dự đoán
                        syn_outputs = self.model(syn_inputs.detach())

                        # b. Tính KL Loss: Student cố gắng khớp distribution của Teacher
                        kd_loss = KD_loss(syn_outputs, syn_old_outputs.detach(), 2)
                        
                        loss_replay = self.args.kd * kd_loss

                # --- 3. Tổng hợp Loss ---
                total_loss = loss_real + loss_replay
                local_loss.append(total_loss.item())
                total_loss.backward()
                self.optimizer.step()
            self.learning_rate_scheduler.step()
        label_list = get_class_counts(trainloader)
        self.unique_labels.update(label_list.keys())
        print(f"[Client-side] ID: {self.id}, task-id: {task} with data distribution {label_list} \nand loss {sum(local_loss)/len(local_loss)} (CE: {loss_real} and Replay: {loss_replay}), lr: {self.optimizer.param_groups[0]['lr']}")
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time