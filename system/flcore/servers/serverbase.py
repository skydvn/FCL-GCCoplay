import os
import json
import shutil
import torch
import torch.nn.functional as F
import wandb
import numpy as np
import csv
import copy
import time
import random
from datetime import datetime
from utils.data_utils import *
from flcore.metrics.average_forgetting import metric_average_forgetting

import statistics

class Server(object):
    def __init__(self, args, times):
        # Set up the main attributes
        self.args = args
        self.device = args.device
        self.dataset = args.dataset
        self.num_classes = args.num_classes
        self.global_rounds = args.global_rounds
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.global_model = copy.deepcopy(args.model)
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        self.num_join_clients = int(self.num_clients * self.join_ratio)
        self.current_num_join_clients = self.num_join_clients
        self.algorithm = args.algorithm
        self.time_threthold = args.time_threthold
        self.offlog = args.offlog
        self.save_folder = f"{args.out_folder}/{args.dataset}_{args.algorithm}_{args.model_str}_{args.optimizer}_lr{args.local_learning_rate}_{args.note}" if args.note else f"{args.out_folder}/{args.dataset}_{args.algorithm}_{args.model_str}_{args.optimizer}_lr{args.local_learning_rate}"
        if self.offlog:    
            if os.path.exists(self.save_folder):
                shutil.rmtree(self.save_folder)
            os.makedirs(self.save_folder, exist_ok=True)

        self.clients = []
        self.selected_clients = []
        self.train_slow_clients = []
        self.send_slow_clients = []

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []

        self.times = times
        self.eval_gap = args.eval_gap
        self.client_drop_rate = args.client_drop_rate

        self.global_accuracy_matrix = []
        self.local_accuracy_matrix = []

        # if self.args.dataset == 'IMAGENET1k':
        #     self.N_TASKS = 500
        # elif self.args.dataset == 'CIFAR100':
        #     self.N_TASKS = 50
        # elif self.args.dataset == 'CIFAR10':
        #     self.N_TASKS = 5
        # if self.args.nt is not None:
        #     self.N_TASKS = self.args.num_classes // self.args.cpt
        self.N_TASKS = self.args.num_tasks
        assert self.args.num_classes // self.args.cpt == self.N_TASKS 
        # FCL
        self.task_dict = {}
        self.current_task = 0

        self.angle_value = 0
        self.grads_angle_value = 0
        self.distance_value = 0
        self.norm_value = 0

        self.file_name = f"{self.args.algorithm}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

        self.client_task_sequences = []
        for i in range(self.num_clients):
            # Tạo danh sách Task ID từ 0 đến N_TASKS-1
            sequence = list(range(self.N_TASKS))
            # Hoán vị ngẫu nhiên để tạo tính Temporal Heterogeneity
            random.shuffle(sequence) 
            self.client_task_sequences.append(sequence)

        # Theo dõi bước hiện tại trong lộ trình (curriculum step)
        self.current_curriculum_step = 0

        # Chia 100 class thành 5 cụm cố định
        all_classes = list(range(self.num_classes))
        # random.shuffle(all_classes) # Bỏ comment nếu muốn task 0 không nhất thiết là class 0-19
        self.global_task_labels = {}
        for t in range(self.N_TASKS):
            self.global_task_labels[t] = all_classes[t*self.args.cpt : (t+1)*self.args.cpt]

    def set_clients(self, clientObj):
        self.global_task_label_mapping = {}
        
        # 1. Tạo Pool dữ liệu cho từng Task (Load 1 lần duy nhất mỗi Task)
        task_data_pools = {}
        print("--- Pre-loading and Shuffling Task Pools ---")
        for tid in range(self.N_TASKS):
            # Load toàn bộ 10.000 ảnh của Task ID này
            # (Dùng client_id=0 hoặc -1 tùy read_func để lấy full task data)
            if self.args.dataset == 'IMAGENET1k':
                read_func = read_client_data_FCL_imagenet1k
            elif self.args.dataset == 'CIFAR100':
                read_func = read_client_data_FCL_cifar100
            elif self.args.dataset == 'CIFAR10':
                read_func = read_client_data_FCL_cifar10
            else:
                raise NotImplementedError("Not supported dataset")
            full_task_data, label_info = read_func(0, task=tid, classes_per_task=self.args.cpt, count_labels=True)
            
            # Lưu trữ data và danh sách index đã xáo trộn
            indices = list(range(len(full_task_data)))
            random.shuffle(indices)
            
            task_data_pools[tid] = {
                'dataset': full_task_data,
                'indices': indices,
                'labels': label_info['labels']
            }
            self.global_task_label_mapping[tid] = label_info['labels']

        # 2. Phân phối dữ liệu cho từng Client
        for i in range(self.num_clients):
            # Xác định Client i đang ở bước đầu tiên của trình tự nào
            initial_task_id = self.client_task_sequences[i][0]
            pool = task_data_pools[initial_task_id]
            
            # 3. Tính toán số mẫu cho client này (Ví dụ: chia đều + nhiễu ngẫu nhiên)
            # Giả sử 10.000 ảnh chia cho 10 client học task đó -> ~1000 ảnh/client
            avg_samples = len(pool['dataset']) // (self.num_clients // self.N_TASKS + 1)
            num_samples = random.randint(int(avg_samples * 0.5), int(avg_samples * 1.5))
            
            # Đảm bảo không lấy quá số lượng còn lại trong pool
            num_samples = min(num_samples, len(pool['indices']))
            
            # Bốc thăm các chỉ số index (Không hoàn lại - No replacement)
            chosen_indices = pool['indices'][:num_samples]
            pool['indices'] = pool['indices'][num_samples:] # Loại bỏ các index đã dùng
            
            # Tạo Subset dataset cho client
            client_train_data = torch.utils.data.Subset(pool['dataset'], chosen_indices)

            # 4. Khởi tạo Client
            client = clientObj(self.args, id=i, train_data=client_train_data, task_sequence=self.client_task_sequences[i])
            client.current_task_id = initial_task_id
            client.task_dict[initial_task_id] = pool['labels']
            client.current_labels = list(pool['labels'])
            client.classes_so_far = list(pool['labels'])
            client.file_name = self.file_name
            
            self.clients.append(client)
            print(f"Client {i:02d} | Task {initial_task_id} | Samples: {len(client_train_data)} | Labels: {pool['labels']}")

        self.print_task_curriculum()

    # def set_clients(self, clientObj):
    #     self.global_task_label_mapping = self.global_task_labels

    #     for i in range(self.num_clients):
    #         print(f"Creating client {i} ...")
    #         # Lấy Task ID thực tế từ trình tự ngẫu nhiên của client i
    #         initial_task_id = self.client_task_sequences[i][0]
    #         target_labels = self.global_task_labels[initial_task_id]
    #         if self.args.dataset == 'IMAGENET1k':
    #             read_func = read_client_data_FCL_imagenet1k
    #         elif self.args.dataset == 'CIFAR100':
    #             read_func = read_client_data_FCL_cifar100
    #         elif self.args.dataset == 'CIFAR10':
    #             read_func = read_client_data_FCL_cifar10
    #         else:
    #             raise NotImplementedError("Not supported dataset")

    #         train_data, label_info = read_func(i, task=initial_task_id, classes_per_task=self.args.cpt, count_labels=True)

    #         # 4. Khởi tạo Client
    #         client = clientObj(self.args, id=i, train_data=train_data, task_sequence=self.client_task_sequences[i])
            
    #         # --- ÉP METADATA THEO CHUẨN SERVER ---
    #         client.current_task_id = initial_task_id
    #         client.file_name = self.file_name
            
    #         # QUAN TRỌNG: Gán nhãn chuẩn từ Server để các client cùng Task ID sẽ dùng chung Output Neurons
    #         # Nếu hàm read_func trả về nhãn sai (trùng), ta ghi đè bằng nhãn chuẩn
    #         actual_labels = list(target_labels) 
            
    #         client.task_dict[initial_task_id] = actual_labels
    #         client.current_labels = actual_labels
    #         client.classes_so_far = actual_labels
            
    #         self.clients.append(client)
    #         print(f"Client {i:02d} | Task {initial_task_id} | Samples: {len(train_data)} | Classes: {len(label_info['labels'])}")

    #     self.print_task_curriculum()

    def select_clients(self):
        if self.random_join_ratio:
            self.current_num_join_clients = np.random.choice(range(self.num_join_clients, self.num_clients+1), 1, replace=False)[0]
        else:
            self.current_num_join_clients = self.num_join_clients
        selected_clients = list(np.random.choice(self.clients, self.current_num_join_clients, replace=False))

        return selected_clients

    def send_models(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()
            
            client.set_parameters(self.global_model)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_clients = sorted(
            random.sample(
                self.selected_clients, 
                int((1 - self.client_drop_rate) * self.current_num_join_clients)
            ), 
            key=lambda client: client.id
        )

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        for client in active_clients:
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                        client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:
                tot_samples += len(client.train_data)
                self.uploaded_ids.append(client.id)
                self.uploaded_weights.append(len(client.train_data))
                self.uploaded_models.append(client.model)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def receive_grads(self):

        self.grads = copy.deepcopy(self.uploaded_models)
        # This for copy the list to store all the gradient update value

        for model in self.grads:
            for param in model.parameters():
                param.data.zero_()

        for grad_model, local_model in zip(self.grads, self.uploaded_models):
            for grad_param, local_param, global_param in zip(grad_model.parameters(), local_model.parameters(),
                                                             self.global_model.parameters()):
                grad_param.data = local_param.data - global_param.data
        for w, client_model in zip(self.uploaded_weights, self.grads):
            self.mul_params(w, client_model)

    def mul_params(self, w, client_model):
        for param in client_model.parameters():
            param.data = param.data.clone() * w

    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data.zero_()
            
        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)

    def add_parameters(self, w, client_model):
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w

    def test_metrics(self, task, glob_iter, flag):
        
        num_samples = []
        tot_correct = []
        for c in self.clients:
            ct, ns = c.test_metrics(task)
            tot_correct.append(ct*1.0)
            num_samples.append(ns)

            test_acc = sum(tot_correct)*1.0 / sum(num_samples)
    
            if flag != "off":
                if flag == "global":
                    subdir = os.path.join(self.save_folder, f"Client_Global/Client_{c.id}")
                    log_key = f"Client_Global/Client_{c.id}/Averaged Test Accurancy"
                elif flag == "local":
                    subdir = os.path.join(self.save_folder, f"Client_Local/Client_{c.id}")
                    log_key = f"Client_Local/Client_{c.id}/Averaged Test Accurancy"

                if self.args.wandb:
                    wandb.log({log_key: test_acc}, step=glob_iter)
                
                if self.offlog:
                    os.makedirs(subdir, exist_ok=True)

                    file_path = os.path.join(subdir, "test_accuracy.csv")
                    file_exists = os.path.isfile(file_path)

                    with open(file_path, mode="w", newline="") as f:
                        writer = csv.writer(f)
                        if not file_exists:
                            writer.writerow(["Step", "Test Accuracy"])  
                        writer.writerow([glob_iter, test_acc]) 

        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct

    def train_metrics(self, task=None):

        num_samples = []
        losses = []
        for c in self.clients:
            cl, ns = c.train_metrics(task=task)
            num_samples.append(ns)
            losses.append(cl*1.0)

        ids = [c.id for c in self.clients]

        return ids, num_samples, losses

    # evaluate selected clients
    def eval(self, task, glob_iter, flag):
        stats = self.test_metrics(task, glob_iter, flag=flag)
        stats_train = self.train_metrics(task=task)
        test_acc = sum(stats[2])*1.0 / sum(stats[1])
        train_loss = sum(stats_train[2])*1.0 / sum(stats_train[1])
        if flag == "global":
            subdir = os.path.join(self.save_folder, "Global")
            log_keys = {
                "Global/Averaged Train Loss": train_loss,
                "Global/Averaged Test Accuracy": test_acc,
                # "Global/Averaged Angle": self.angle_value,
                "Global/Averaged Grads Angle": self.grads_angle_value,
                "Global/Averaged Distance": self.distance_value,
                "Global/Averaged GradNorm": self.norm_value,
            }
            if self.args.tgm:
                self.t_angle_after = statistics.mean(client.t_angle_after for client in self.selected_clients)

                log_keys.update({
                    "Global/Timestep Angle After": self.t_angle_after,
                })
                # print(log_keys)

        elif flag == "local":
            subdir = os.path.join(self.save_folder, "Local")
            log_keys = {
                "Local/Averaged Train Loss": train_loss,
                "Local/Averaged Test Accuracy": test_acc,
            }

        if self.args.log and flag == "global":
            # print(f"{sum(stats_train[2])}, {sum(stats_train[1])}")task_id
            print(f"Global Averaged Test Accuracy: {test_acc}")
            print(f"Global Averaged Test Loss: {train_loss}")

        if self.args.log and flag == "local":
            # print(f"{sum(stats_train[2])}, {sum(stats_train[1])}")
            print(f"Local Averaged Test Accuracy: {test_acc}")
            print(f"Local Averaged Test Loss: {train_loss}")

        if self.args.wandb:
            wandb.log(log_keys, step=glob_iter)

        if self.offlog:
            os.makedirs(subdir, exist_ok=True)

            file_path = os.path.join(subdir, "metrics.csv")
            file_exists = os.path.isfile(file_path)

            with open(file_path, mode="a", newline="") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["Step", "Train Loss", "Test Accuracy"])  
                writer.writerow([glob_iter, train_loss, test_acc]) 

    # evaluate after end 1 task
    def eval_task(self, task, glob_iter, flag):
        accuracy_on_all_task = []

        for t in range(self.N_TASKS):
            stats = self.test_metrics(task=t, glob_iter=glob_iter, flag="off")
            test_acc = sum(stats[2]) * 1.0 / sum(stats[1])
            accuracy_on_all_task.append(test_acc)

        if flag == "global":
            self.global_accuracy_matrix.append(accuracy_on_all_task)
            accuracy_matrix = self.global_accuracy_matrix
            subdir = os.path.join(self.save_folder, "Global")
            log_key = "Global/Averaged Forgetting"
        elif flag == "local":
            self.local_accuracy_matrix.append(accuracy_on_all_task)
            accuracy_matrix = self.local_accuracy_matrix
            subdir = os.path.join(self.save_folder, "Local")
            log_key = "Local/Averaged Forgetting"

        forgetting = metric_average_forgetting(int(task%self.N_TASKS), accuracy_matrix)

        if self.args.wandb:
            wandb.log({log_key: forgetting}, step=glob_iter)

        print(f"{log_key}: {forgetting:.4f}")

        if self.offlog:
            os.makedirs(subdir, exist_ok=True)

            csv_filename = os.path.join(subdir, f"{self.args.algorithm}_accuracy_matrix.csv")
            with open(csv_filename, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerows(accuracy_matrix)

    def assign_unique_tasks(self):
        # Convert lists to sets of tuples for easy comparison
        unique_set = {tuple(task) for task in self.unique_task}
        old_unique_set = {tuple(task) for task in self.old_unique_task}

        # Find new tasks by taking the difference
        new_tasks = unique_set - old_unique_set
        # print(f"new_tasks: {new_tasks}")
        # Loop over new tasks and assign them to task_dict
        for task in new_tasks:
            self.current_task += 1
            self.task_dict[self.current_task] = list(task)

    def cos_sim(self, prev_model, model1, model2):
        prev_param = torch.cat([p.data.view(-1) for p in prev_model.parameters()])
        params1 = torch.cat([p.data.view(-1) for p in model1.parameters()])
        params2 = torch.cat([p.data.view(-1) for p in model2.parameters()])

        grad1 = params1 - prev_param
        grad2 = params2 - prev_param

        cos_sim = torch.dot(grad1, grad2) / (torch.norm(grad1) * torch.norm(grad2))
        return cos_sim.item()

    def cosine_similarity(self, model1, model2):
        params1 = torch.cat([p.data.view(-1) for p in model1.parameters()])
        params2 = torch.cat([p.data.view(-1) for p in model2.parameters()])
        cos_sim = torch.dot(params1, params2) / (torch.norm(params1) * torch.norm(params2))
        return cos_sim.item()

    def distance(self, model1, model2):
        params1 = torch.cat([p.data.view(-1) for p in model1.parameters()])
        params2 = torch.cat([p.data.view(-1) for p in model2.parameters()])

        mse = F.mse_loss(params1, params2)
        return mse.item()

    def spatio_grad_eval(self, model_origin, glob_iter):
        angle = [self.cos_sim(model_origin, self.global_model, models) for models in self.uploaded_models]
        distance = [self.distance(self.global_model, models) for models in self.uploaded_models]
        norm = [self.distance(model_origin, models) for models in self.uploaded_models]
        self.angle_value = statistics.mean(angle)
        self.distance_value = statistics.mean(distance)
        self.norm_value = statistics.mean(norm)
        angle_value = []

        # for grad_i in self.grads:
        #     for grad_j in self.grads:
        #         angle_value.append(self.cosine_similarity(grad_i, grad_j))

        for i in range(len(self.grads)):
            for j in range(i + 1, len(self.grads)):
                angle_value.append(self.cosine_similarity(self.grads[i], self.grads[j]))

        cosine_to_client0 = {}
        count_positive = 0  # cosine > 0
        count_negative = 0  # cosine >= 0

        for i in range(1, len(self.grads)):
            sim = self.cosine_similarity(self.grads[0], self.grads[i])
            cosine_to_client0[f"{i}"] = sim

            if sim > 0:
                count_positive += 1
            if sim <= 0:
                count_negative += 1

        if self.args.wandb:
            wandb.log({f"cosine/{k}": v for k, v in cosine_to_client0.items()}, step=glob_iter)

            wandb.log({
                "cosine_count/positive (>0)": count_positive,
                "cosine_count/negative (<=0)": count_negative
            }, step=glob_iter)

        self.grads_angle_value = statistics.mean(angle_value)
        # print(f"grad angle: {self.grads_angle_value}")

    def proto_eval(self, global_model, local_model, task, round):
        # TODO save models to ./pca_eval/file_name/global
        model_filename = f"task_{task}_round_{round}.pth"
        save_dir = os.path.join("pca_eval", self.file_name, "global")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Save model state_dict
        model_path = os.path.join(save_dir, model_filename)
        torch.save(global_model.state_dict(), model_path)

        save_dir = os.path.join("pca_eval", self.file_name, "local")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Save model state_dict
        model_path = os.path.join(save_dir, model_filename)
        torch.save(local_model.state_dict(), model_path)

    def print_task_curriculum(self):
        print("\n" + "="*60)
        print("CLIENT TASK CURRICULUM (Temporal Heterogeneity)")
        print("="*60)
        
        # In trình tự của 5 client đầu tiên (để tránh spam nếu có 100 client)
        for i in range(self.num_clients):
            seq = self.client_task_sequences[i]
            seq_str = " -> ".join([f"Task {tid}" for tid in seq])
            print(f"Client {i:02d}: {seq_str}")
        
        print("\n" + "="*60)
        print("TASK ID TO LABEL MAPPING (Spatial Heterogeneity Check)")
        print("="*60)
        
        # Sắp xếp theo Task ID (0, 1, 2, 3, 4)
        for tid in sorted(self.global_task_label_mapping.keys()):
            labels = self.global_task_label_mapping[tid]
            num_labels = len(labels)
            
                
            print(f"Task ID {tid}: Total {num_labels:2d} classes | Preview: {labels}")
            
            # Kiểm tra nhanh: Nếu không phải 20 classes thì cảnh báo ngay
            if num_labels != self.args.cpt:
                print(f"   [!] WARNING: Task {tid} has {num_labels} classes, expected {self.args.cpt}!")

        print("="*60 + "\n")