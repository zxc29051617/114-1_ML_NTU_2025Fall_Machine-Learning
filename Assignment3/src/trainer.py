import torch
from torch import nn
from tqdm import tqdm
import math

from src.utils import dict_to_device

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
        eval_func: callable,
        train_loader: torch.utils.data.DataLoader,
        valid_loader: torch.utils.data.DataLoader,
        save_path: str = "best_model.pth",
        logger = None,
    ):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.eval_func = eval_func
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.save_path = save_path
        self.logger = logger

        # 自動偵測：判斷評估指標是「越小越好」(Minimize) 還是「越大越好」(Maximize)
        # 如果 eval_func 的類別名稱包含 "MSE" 或 "Loss"，我們假設它是 Loss，越小越好。
        # 否則 (如 DiceScore)，我們假設它是 Score，越大越好。
        self.eval_name = self.eval_func.__class__.__name__
        self.is_minimize = "MSE" in self.eval_name or "Loss" in self.eval_name

        # 初始化最佳分數
        # 如果是找最小值，初始分數設為無限大；如果是找最大值，設為 0 (或負無限大)
        if self.is_minimize:
            self.best_score = float('inf')
        else:
            self.best_score = float('-inf') 

    def train_one_epoch(self):
        self.model.train()
        train_loss = 0.0
        
        self.optimizer.zero_grad() 
        
        for i, batch_data in enumerate(tqdm(self.train_loader, desc="Training")): 
            batch_data = dict_to_device(batch_data, self.device)
            inputs = batch_data["image"]
            labels = batch_data["label"]
            
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            loss = loss / self.accumulation_steps
            loss.backward()

            if (i + 1) % self.accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.lr_scheduler.step()
            
            loss_item = loss.item() * self.accumulation_steps # 還原 loss數值以便記錄
            train_loss += loss_item
            
            if self.logger:
                self.logger.log({"train/loss": loss_item})
                self.logger.log({"train/lr": self.lr_scheduler.get_last_lr()[0]})
        
        avg_train_loss = train_loss / len(self.train_loader)
        if self.logger:
            self.logger.log({"train/epoch_loss": avg_train_loss})

    def valid_one_epoch(self):
        scores = []
        self.model.eval()
        
        with torch.inference_mode():
            for batch_data in tqdm(self.valid_loader, desc="Validation"):
                batch_data = dict_to_device(batch_data, self.device)
                
                inputs = batch_data["image"]
                labels = batch_data["label"]
                
                outputs = self.model(inputs)

                # 呼叫評估函數
                # 注意：如果你的 MSELoss 不接受 mean=True，記得在 src/loss.py 加 **kwargs
                score = self.eval_func(outputs, labels, mean=True)
                
                scores.append(score.item())

        avg_score = sum(scores) / len(scores)
        
        print(f"Validation Metric ({self.eval_name}): {avg_score:.4f}")
        if self.logger:
            self.logger.log({f"valid/{self.eval_name}": avg_score})
        return avg_score

    def fit(self, n_epoch, accumulation_steps=1):
        self.model.to(self.device)
        self.accumulation_steps = accumulation_steps
        
        for epoch in range(n_epoch):
            print(f"Epoch {epoch+1}/{n_epoch}")
            self.train_one_epoch()
            valid_score = self.valid_one_epoch()

            # 判斷是否存檔
            is_better = False
            if self.is_minimize:
                # 如果是 MSE (越小越好)
                if valid_score < self.best_score:
                    is_better = True
            else:
                # 如果是 Dice (越大越好)
                if valid_score > self.best_score:
                    is_better = True

            if is_better:
                self.best_score = valid_score
                torch.save(self.model.state_dict(), self.save_path)
                print(f"Successfully saved best model with score: {self.best_score:.4f}")
            else:
                print(f"Score did not improve (Best: {self.best_score:.4f})")