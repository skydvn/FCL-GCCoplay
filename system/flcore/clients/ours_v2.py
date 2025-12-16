import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import copy
from flcore.clients.clientbase import Client
from flcore.utils_core.target_utils import Generator
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, img_size=32, nc=3, num_classes=100, device=None):
        super(Generator, self).__init__()
        
        # 1. Update params to include num_classes
        self.params = (nz, ngf, img_size, nc, num_classes) 
        self.init_size = img_size // 4
        self.device = device
        
        # 2. Add Label Embedding
        # We embed the label into a vector of size 'nz' (same as noise)
        self.label_emb = nn.Embedding(num_classes, nz)

        # 3. Update First Linear Layer
        # Input size is now nz (noise) + nz (label_embedding) = nz * 2
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

    # 4. Update Forward to accept labels
    def forward(self, z, labels):
        # Generate label embeddings
        c = self.label_emb(labels)
        
        # Concatenate noise (z) and label embeddings (c)
        # z shape: [batch, nz], c shape: [batch, nz] -> input shape: [batch, nz*2]
        gen_input = torch.cat((z, c), -1)
        
        out = self.l1(gen_input)
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

    def clone(self):
        # 5. Update clone to pass num_classes (self.params[4])
        clone = Generator(self.params[0], self.params[1], self.params[2], self.params[3], self.params[4], device=self.device)
        clone.load_state_dict(self.state_dict())
        return clone.to(self.device)

class clientOursV2(Client):
    def __init__(self, args, id, train_data, **kwargs):
        super().__init__(args, id, train_data, **kwargs)
        # Cấu hình Generator
        if 'cifar' in args.dataset.lower():
            self.img_size = 32; self.nz = 256
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
        
        # 
        # Định nghĩa KL Loss function
        criterion_kd = nn.CrossEntropyLoss()

        start_time = time.time()
        max_local_epochs = self.local_epochs

        for epoch in range(max_local_epochs):
            for i, (x_real, y_real) in enumerate(trainloader):
                if isinstance(x_real, list): x_real = x_real[0]
                x_real = x_real.to(self.device)
                y_real = y_real.to(self.device)

                # --- 1. Real Data: CrossEntropy Loss ---
                output_real = self.model(x_real)
                loss_real = self.loss(output_real, y_real)

                # --- 2. Replay Data: KL Divergence Loss (Distillation) ---
                loss_replay = 0
                # Chỉ Replay khi có Teacher (old_network) và task > 0
                if self.current_task > 0 and self.old_network is not None:
                    batch_size = x_real.shape[0]
                    
                    # a. Sinh dữ liệu giả
                    z = torch.randn(batch_size, self.nz).to(self.device)
                    # Chọn nhãn từ các task CŨ để ôn tập
                    # Lưu ý: Teacher chỉ biết kiến thức cũ
                    if len(self.classes_past_task) > 0:
                        fake_labels = np.random.choice(self.classes_past_task, batch_size)
                        fake_labels = torch.tensor(fake_labels).long().to(self.device)
                        
                        with torch.no_grad():
                            x_fake = self.generator(z, fake_labels)
                            # Teacher (Old Model) dự đoán soft labels
                            teacher_logits = self.old_network(x_fake)

                        # Student (Current Model) dự đoán
                        student_logits = self.model(x_fake.detach())

                        # b. Tính KL Loss: Student cố gắng khớp distribution của Teacher
                        # Công thức: KL(log_softmax(S/T), softmax(T/T)) * T^2
                        loss_kd = criterion_kd(
                            F.log_softmax(student_logits / self.T, dim=1),
                            F.softmax(teacher_logits / self.T, dim=1)
                        ) * (self.T * self.T)
                        
                        loss_replay = loss_kd

                # --- 3. Tổng hợp Loss ---
                total_loss = loss_real + self.replay_weight * loss_replay
                
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time