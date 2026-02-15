import time
import math
import sys
import os
import random

# --- Import C++ Tensor ---
try:
    from my_framework_cpp import Tensor
except ImportError:
    print("Run: python3 setup.py build_ext --inplace")
    sys.exit(1)

from data import Dataset, DataLoader
from layers import Conv2d, Linear, ReLU, MaxPool2d, Flatten

# --------------------------------------------------
# Helper: Stack batch tensors
# --------------------------------------------------
def stack_tensors(tensor_list):
    if not tensor_list:
        return None
    shape = tensor_list[0].shape
    N = len(tensor_list)
    new_shape = [N] + shape
    combined_data = []
    for t in tensor_list:
        combined_data.extend(t.data)
    return Tensor(combined_data, new_shape)

# --------------------------------------------------
# Subset wrapper
# --------------------------------------------------
class SubsetDataset:
    def __init__(self, data_list):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# --------------------------------------------------
# CNN MODEL (Single Conv - Best Version)
# --------------------------------------------------
class SimpleCNN:
    def __init__(self, num_classes=100):
        # 32x32 -> Conv(5x5, pad=2) -> 32x32 -> Pool -> 16x16
        self.conv1 = Conv2d(in_channels=3, out_channels=16,
                            kernel_size=5, stride=1, padding=2)

        self.relu = ReLU()
        self.pool = MaxPool2d(kernel_size=2, stride=2)
        self.flat = Flatten()

        # 16 * 16 * 16 = 4096 features
        self.fc1 = Linear(in_features=16 * 16 * 16,
                          out_features=num_classes)

    def forward(self, x):
        self.x = x
        self.conv_out = self.conv1.forward(self.x)
        self.relu_out = self.relu.forward(self.conv_out)
        self.pool_out = self.pool.forward(self.relu_out)

        N, H, W, C = self.pool_out.shape
        self.flat_out = Tensor(self.pool_out.data, [N, H * W * C])

        self.logits = self.fc1.forward(self.flat_out)
        return self.logits

    def backward(self, grad_logits, lr=0.01):
        batch_size = grad_logits.shape[0]
        num_classes = grad_logits.shape[1]

        # FC backward
        grad_fc_weights = self.flat_out.matmul_transpose_left(grad_logits)
        fc_weights_T = self.fc1.weight.transpose()
        grad_fc_input = grad_logits.matmul(fc_weights_T)

        # Reshape
        grad_pool_out = Tensor(grad_fc_input.data, self.pool_out.shape)

        # Pool backward
        grad_relu_out = self.pool.maxpool2d_backward(
            self.relu_out, grad_pool_out, 2, 2
        )

        # ReLU backward
        grad_conv_out = self.conv_out.relu_backward(grad_relu_out)

        # Conv backward
        grad_conv_weights = self.x.conv2d_grad_weight(
            self.x, grad_conv_out, 5, 1, 2
        )

        # Bias grads
        bias_grads = grad_logits.sum_axis0(batch_size, num_classes)

        # SGD update
        self.fc1.weight.sgd_update(grad_fc_weights, lr)
        self.fc1.bias.sgd_update(bias_grads, lr)
        self.conv1.weight.sgd_update(grad_conv_weights, lr)

    def get_summary(self):
        print(f"Model: 32x32 -> Conv(16 filters, 5x5) -> ReLU -> Pool -> Linear({self.fc1.out_features})")

# --------------------------------------------------
# Cross Entropy Loss (Stable Softmax)
# --------------------------------------------------
def cross_entropy_loss(logits, targets):
    loss_sum = 0.0
    batch_size = logits.shape[0]
    num_classes = logits.shape[1]
    grad_data = [0.0] * (batch_size * num_classes)

    for i in range(batch_size):
        start = i * num_classes
        scores = logits.data[start:start + num_classes]

        max_score = max(scores)
        exps = [math.exp(s - max_score) for s in scores]
        sum_exps = sum(exps)
        probs = [e / sum_exps for e in exps]

        true_class = targets[i]
        p_clamped = max(probs[true_class], 1e-12)
        loss_sum -= math.log(p_clamped)

        for c in range(num_classes):
            grad = probs[c]
            if c == true_class:
                grad -= 1.0
            grad_data[start + c] = grad / batch_size

    return (loss_sum / batch_size), Tensor(grad_data, logits.shape)

# --------------------------------------------------
# Accuracy
# --------------------------------------------------
def calculate_accuracy(logits, targets):
    correct = 0
    batch_size = logits.shape[0]
    num_classes = logits.shape[1]

    for i in range(batch_size):
        start = i * num_classes
        scores = logits.data[start:start + num_classes]
        pred = scores.index(max(scores))
        if pred == targets[i]:
            correct += 1

    return correct

# --------------------------------------------------
# MAIN
# --------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.005)
    args = parser.parse_args()

    random.seed(42)

    dataset_name = args.data_path
    BATCH_SIZE = args.batch_size
    LR = args.lr
    EPOCHS = args.epochs

    full_dataset = Dataset(dataset_name)
    all_data = full_dataset.data if hasattr(full_dataset, 'data') else full_dataset.samples

    random.shuffle(all_data)

    split_idx = int(0.8 * len(all_data))
    train_dataset = SubsetDataset(all_data[:split_idx])
    eval_dataset = SubsetDataset(all_data[split_idx:])

    print(f"Train: {len(train_dataset)} | Eval: {len(eval_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False)

    if hasattr(full_dataset, "class_names"):
        num_classes = len(full_dataset.class_names)
    else:
        num_classes = 100 if "data_2" in dataset_name else 10

    model = SimpleCNN(num_classes=num_classes)
    model.get_summary()

    print("Starting training...")
    start_time = time.time()

    for epoch in range(EPOCHS):
        epoch_loss, epoch_acc, total_samples = 0.0, 0, 0

        for tensors, labels in train_loader:
            x_batch = stack_tensors(tensors)
            logits = model.forward(x_batch)
            loss, grad_logits = cross_entropy_loss(logits, labels)
            model.backward(grad_logits, lr=LR)

            epoch_loss += loss * len(labels)
            epoch_acc += calculate_accuracy(logits, labels)
            total_samples += len(labels)

        eval_acc, eval_samples = 0, 0
        for tensors, labels in eval_loader:
            x_batch = stack_tensors(tensors)
            logits = model.forward(x_batch)
            eval_acc += calculate_accuracy(logits, labels)
            eval_samples += len(labels)

        print(f"Epoch {epoch+1} | Loss: {epoch_loss/total_samples:.4f} | "
              f"Train Acc: {epoch_acc/total_samples:.2%} | "
              f"Val Acc: {eval_acc/eval_samples:.2%}")

    print(f"\nTraining finished in {time.time() - start_time:.2f} seconds.")