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


        # Theo dõi bước hiện tại trong lộ trình (curriculum step)
        self.current_curriculum_step = 0

        # Chia 100 class thành 5 cụm cố định
        all_classes = list(range(self.num_classes))
        # random.shuffle(all_classes) # Bỏ comment nếu muốn task 0 không nhất thiết là class 0-19
        self.global_task_labels = {}
        for t in range(self.N_TASKS):
            self.global_task_labels[t] = all_classes[t*self.args.cpt : (t+1)*self.args.cpt]

        # FCL Metric Histories
        self.kr_t_history = []
        self.kr_s_history = []
        self.global_accuracy_matrix = []

    def set_clients(self, clientObj):
        for i in range(self.num_clients):
            print(f"Creating client {i} ...")
            if self.args.dataset == 'IMAGENET1k':
                train_data, label_info = read_client_data_FCL_imagenet1k(i, task=0, classes_per_task=self.args.cpt, count_labels=True)
            elif self.args.dataset == 'CIFAR100':
                train_data, label_info = read_client_data_FCL_cifar100(i, task=0, classes_per_task=self.args.cpt, count_labels=True)
            elif self.args.dataset == 'CIFAR10':
                train_data, label_info = read_client_data_FCL_cifar10(i, task=0, classes_per_task=self.args.cpt, count_labels=True)
            else:
                raise NotImplementedError("Not supported dataset")

            client = clientObj(self.args, id=i, train_data=train_data)
            self.clients.append(client)

            # update classes so far & current labels
            client.classes_so_far.extend(label_info['labels'])
            client.current_labels.extend(label_info['labels'])
            client.task_dict[0] = label_info['labels']
            client.file_name = self.file_name
        self.print_task_curriculum()

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
            ct, ns = c.test_metrics(task=task)
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
                "Global/Averaged Angle": self.angle_value,
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

    def spatio_grad_eval(self, model_origin):
        angle = [self.cos_sim(model_origin, self.global_model, models) for models in self.uploaded_models]
        distance = [self.distance(self.global_model, models) for models in self.uploaded_models]
        norm = [self.distance(model_origin, models) for models in self.uploaded_models]
        self.angle_value = statistics.mean(angle)
        self.distance_value = statistics.mean(distance)
        self.norm_value = statistics.mean(norm)
        angle_value = []
        for grad_i in self.grads:
            for grad_j in self.grads:
                angle_value.append(self.cosine_similarity(grad_i, grad_j))
        self.grads_angle_value = statistics.mean(angle_value)
        print(f"grad angle: {self.grads_angle_value}")

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
        print(f"TASK CURRICULUM REPORT (Total Clients: {len(self.clients)})")
        print("="*60)

        for client in self.clients:
            print(f"Client ID: {client.id}")
            
            # Check if the dictionary is empty
            if not client.task_dict:
                print("   [No tasks assigned yet]")
                continue

            # Sort task IDs to print in order (Task 0, Task 1...)
            sorted_task_ids = sorted(client.task_dict.keys())
            
            for t_id in sorted_task_ids:
                labels = client.task_dict[t_id]
                
                # Handle different data types (Tensor, Numpy, List)
                if hasattr(labels, 'tolist'):
                    labels = labels.tolist()
                elif isinstance(labels, list):
                    pass # It's already a list
                else:
                    # If it's a single number or unexpected type
                    labels = list(labels)

                # Sort labels for easier visual comparison
                labels = sorted(labels)
                
                print(f"   Task {t_id:<2} | Count: {len(labels):<2} | Labels: {labels}")
            
            print("-" * 30)
        print("="*60 + "\n")

    def compute_knowledge_retention(self, current_task, round=None):
        """
        Computes accuracy on all tasks seen so far (0 to current_task) 
        and logs the results to WandB.
        """
        task_accuracies = []
        wandb_logs = {}

        print(f"\n--- Computing Knowledge Retention (Tasks 0 to {current_task}) ---")

        # 1. Iterate through all previous tasks up to the current one
        for task_id in range(current_task + 1):
            total_correct = 0
            total_samples = 0

            # 2. Collect metrics from all clients
            for client in self.clients:
                # OPTIONAL: Sync global model if evaluating global performance
                # client.set_parameters(self.global_model) 

                correct, num_samples = client.test_metrics(task=task_id)
                total_correct += correct
                total_samples += num_samples

            # 3. Calculate accuracy
            if total_samples > 0:
                task_acc = total_correct / total_samples
            else:
                task_acc = 0.0
            
            task_accuracies.append(task_acc)
            
            # Prepare individual task log
            wandb_logs[f"Retention/Task_{task_id}_Acc"] = task_acc
            print(f"   Task {task_id} Accuracy: {task_acc:.4f}")

        # 4. Calculate Average Retention
        avg_retention = sum(task_accuracies) / len(task_accuracies) if task_accuracies else 0.0
        wandb_logs[f"Retention/Average_Acc"] = avg_retention
        
        # 5. Add 'round' or 'task' to x-axis context if provided
        if round is not None:
            wandb_logs['Round'] = round
        wandb_logs['Current_Task'] = current_task

        # 6. Upload to WandB
        if wandb.run is not None:
            wandb.log(wandb_logs)
        
        print(f"   [Summary] Average Retention: {avg_retention:.4f}")
        print("------------------------------------------------------------\n")

        return avg_retention, task_accuracies

    def calculate_table1_accuracy(self, current_step, glob_iter):
        task_accs = {}
        for t_id in range(self.N_TASKS):
            t_corr, t_samp = 0, 0
            for c in self.clients:
                # Evaluation of Global Model on specific Task ID
                ct, ns = c.test_metrics(task=t_id)
                t_corr += ct
                t_samp += ns
            
            acc = (t_corr / t_samp) * 100 if t_samp > 0 else 0
            task_accs[f"Table1/TaskID_{t_id}"] = acc
        
        if self.args.wandb:
            wandb.log(task_accs, step=glob_iter)

    def visualize_tsne(self, step_idx, glob_iter):
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt

        all_feats, all_labels = [], []
        for client in self.clients[:3]: # Lấy mẫu từ 3 client đại diện
            feats, labels = client.get_feature_embeddings(self.global_model)
            all_feats.append(feats)
            all_labels.append(labels)
        
        feats_np = torch.cat(all_feats).cpu().numpy()
        labels_np = torch.cat(all_labels).cpu().numpy()

        tsne = TSNE(n_components=2, random_state=42)
        embeds = tsne.fit_transform(feats_np)

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(embeds[:, 0], embeds[:, 1], c=labels_np, cmap='tab10', s=10)
        plt.colorbar(scatter)
        plt.title(f"t-SNE After Task {step_idx+1}")
        
        if self.args.wandb:
            wandb.log({f"Visual/tSNE_Task_{step_idx+1}": wandb.Image(plt)}, step=glob_iter)
        plt.close()

    def change_task(self, step, glob_iter):
        # Chỉ số 1: Table 1 - Accuracy Matrix (Global Model trên từng Task ID)
        self.calculate_table1_accuracy(step, glob_iter)
        
        # Chỉ số 2 & 3: Temporal và Spatial Knowledge Retention
        self.compute_knowledge_retention(glob_iter)
        
        # Chỉ số 4: Trực quan hóa t-SNE
        self.visualize_tsne(step, glob_iter)