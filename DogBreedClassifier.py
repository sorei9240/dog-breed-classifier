import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision.models import resnet50, ResNet50_Weights
from torch.amp import autocast
from torch.amp import GradScaler

class DogBreedClassifier:
    class AdamWGC(optim.AdamW):
        def step(self, closure=None):
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None or p.dim() <= 1:
                        continue
                    p.grad.data.add_(-p.grad.data.mean(dim=tuple(range(1, p.grad.dim())), keepdim=True))
            return super().step(closure)

    def __init__(self, num_classes=120, batch_size=32):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.scaler = torch.amp.GradScaler('cuda')

        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(60),
            transforms.ColorJitter(brightness=0.7, contrast=0.7, saturation=0.7, hue=0.3),
            transforms.RandomAffine(degrees=30, translate=(0.3, 0.3), scale=(0.7, 1.3), shear=20),
            transforms.RandomPerspective(distortion_scale=0.6, p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.4, scale=(0.02, 0.4))
        ])

        self.val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        num_features = self.model.fc.in_features

        self.model.avgpool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Dropout2d(0.2)
        )

        class ResidualBlock(nn.Module):
            def __init__(self, in_features, out_features):
                super().__init__()
                self.block = nn.Sequential(
                    nn.Linear(in_features, out_features),
                    nn.ReLU(),
                    nn.BatchNorm1d(out_features),
                    nn.Dropout(0.5)
                )
                self.skip = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()

            def forward(self, x):
                return self.block(x) + self.skip(x)

        self.model.fc = nn.Sequential(
            ResidualBlock(num_features, 2048),
            ResidualBlock(2048, 1024),
            nn.Linear(1024, num_classes)
        )

        self.model = self.model.to(self.device)

        self.ema_model = self._create_ema_model()
        self.ema_decay = 0.999

        class FocalCrossEntropyLoss(nn.Module):
            def __init__(self, gamma=2.0, label_smoothing=0.1):
                super().__init__()
                self.gamma = gamma
                self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing, reduction='none')

            def forward(self, inputs, targets):
                ce_loss = self.criterion(inputs, targets)
                pt = torch.exp(-ce_loss)
                focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
                return focal_loss

        self.criterion = FocalCrossEntropyLoss(gamma=2.0, label_smoothing=0.1)

        self.optimizer = self.AdamWGC(
            self.model.parameters(),
            lr=0.0001,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )

    def _create_ema_model(self):
        ema_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        ema_model.fc = self.model.fc
        ema_model.load_state_dict(self.model.state_dict())
        ema_model.to(self.device)
        return ema_model

    def update_ema_model(self):
        with torch.no_grad():
            for ema_param, model_param in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_param.data.mul_(self.ema_decay).add_(model_param.data, alpha=1 - self.ema_decay)

    def load_data(self, data_dir, val_split=0.2):
        dataset = ImageFolder(data_dir, transform=self.train_transform)
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size

        train_dataset, val_dataset = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

        val_dataset.dataset.transform = self.val_transform

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )

        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=0.0001,
            epochs=50,
            steps_per_epoch=len(self.train_loader),
            pct_start=0.4,
            anneal_strategy='cos',
            div_factor=50.0,
            final_div_factor=2000.0
        )

        return len(dataset.classes)

    def mixup_data(self, x, y, alpha=0.4):
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(self.device)

        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in self.train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            if np.random.random() > 0.5:
                inputs, targets_a, targets_b, lam = self.mixup_data(inputs, labels)
                mixed_training = True
            else:
                targets_a = labels
                mixed_training = False

            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda'):
                outputs = self.model(inputs)
                if mixed_training:
                    loss = lam * self.criterion(outputs, targets_a) + (1 - lam) * self.criterion(outputs, targets_b)
                else:
                    loss = self.criterion(outputs, targets_a)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            self.update_ema_model()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)

            if mixed_training:
                correct += (lam * predicted.eq(targets_a).float() +
                          (1 - lam) * predicted.eq(targets_b).float()).sum().item()
            else:
                correct += predicted.eq(targets_a).sum().item()

        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc

    def validate(self):
        self.ema_model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad(), torch.amp.autocast('cuda'):
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.ema_model(inputs)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_loss = running_loss / len(self.val_loader)
        val_acc = 100. * correct / total
        return val_loss, val_acc

    def predict_with_tta(self, inputs, num_augments=15):
        self.ema_model.eval()
        batch_size = inputs.size(0)
        predictions = torch.zeros(batch_size, self.num_classes).to(self.device)

        with torch.no_grad():
            predictions += self.ema_model(inputs).softmax(dim=1)

            for _ in range(num_augments - 1):
                aug_inputs = self.train_transform(inputs.cpu()).to(self.device)
                predictions += self.ema_model(aug_inputs).softmax(dim=1)

        return predictions / num_augments

    def train(self, num_epochs=50, patience=15):
        best_val_acc = 0.0
        patience_counter = 0
        torch.cuda.empty_cache()

        for epoch in range(num_epochs):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()

            print(f'Epoch [{epoch+1}/{num_epochs}]')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            print(f'Learning Rate: {self.scheduler.get_last_lr()[0]:.6f}')

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.ema_model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scaler_state_dict': self.scaler.state_dict(),
                    'best_acc': best_val_acc,
                }, 'best_model.pth')
                print(f'Saved model with validation accuracy: {val_acc:.2f}%')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'Early stopping triggered after {epoch + 1} epochs')
                    break
            print('-' * 60)

classifier = DogBreedClassifier(num_classes=120, batch_size=32)
num_classes = classifier.load_data('images/Images', val_split=0.2)
classifier.train(num_epochs=50, patience=15)
