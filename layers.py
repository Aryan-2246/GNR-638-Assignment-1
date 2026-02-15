import math
import random
from my_framework_cpp import Tensor

class Module:
    def __call__(self, x):
        return self.forward(x)
    
    def get_stats(self):
        # Returns (params, macs, flops)
        return 0, 0, 0

class Linear(Module):
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize Weights (Xavier Initialization approximation)
        limit = math.sqrt(6 / (in_features + out_features))
        w_data = [random.uniform(-limit, limit) for _ in range(in_features * out_features)]
        b_data = [0.0] * out_features
        
        # Weights: [in, out]
        self.weight = Tensor(w_data, [in_features, out_features])
        # Bias: [1, out]
        self.bias = Tensor(b_data, [1, out_features])
        
    def forward(self, x):
        # Y = X @ W + B
        out = x.matmul(self.weight)
        out = out.add(self.bias)
        return out

    def get_stats(self):
        params = (self.in_features * self.out_features) + self.out_features
        # MACs = Input * Output
        macs = self.in_features * self.out_features
        flops = 2 * macs
        return params, macs, flops

class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Initialize Kernels: [K, K, Cin, Cout] (NHWC compatible)
        fan_in = in_channels * kernel_size * kernel_size
        limit = math.sqrt(6 / fan_in)
        total_weights = kernel_size * kernel_size * in_channels * out_channels
        
        w_data = [random.uniform(-limit, limit) for _ in range(total_weights)]
        self.weight = Tensor(w_data, [kernel_size, kernel_size, in_channels, out_channels])
        
        # Bias: [1, Cout] - Broadcasted later
        b_data = [0.0] * out_channels
        self.bias = Tensor(b_data, [1, out_channels])

    def forward(self, x):
        # x: [N, H, W, C]
        out = x.conv2d(self.weight, self.stride, self.padding)
        out = out.add(self.bias)
        return out
        
    def get_stats(self):
        # Weights + Bias
        params = (self.kernel_size**2 * self.in_channels * self.out_channels) + self.out_channels
        # MACs calculation depends on output size, usually estimated per pixel
        # Standard: K*K*Cin * Hout*Wout * Cout
        # We assume 32x32 input -> 32x32 output for stride 1, pad 1
        output_pixels = 32 * 32 
        macs_per_pixel = (self.kernel_size ** 2) * self.in_channels * self.out_channels
        macs = macs_per_pixel * output_pixels
        flops = 2 * macs
        return params, macs, flops

class ReLU(Module):
    def forward(self, x):
        return x.relu()
    
    def get_stats(self):
        return 0, 0, 0

class MaxPool2d(Module):
    def __init__(self, kernel_size=2, stride=2):
        self.kernel_size = kernel_size
        self.stride = stride
        
    def forward(self, x):
        return x.maxpool2d(self.kernel_size, self.stride)

    # --- THE FIX: ADD THIS METHOD ---
    def maxpool2d_backward(self, input_tensor, grad_output, kernel_size, stride):
        # Delegate call to the Tensor's C++ method
        # We can call it on the input_tensor itself
        return input_tensor.maxpool2d_backward(input_tensor, grad_output, kernel_size, stride)
    
    def get_stats(self):
        return 0, 0, 0

class Flatten(Module):
    def forward(self, x):
        # Reshape [N, H, W, C] -> [N, H*W*C]
        # Our Tensor class treats data as flat 1D vector anyway,
        # so we just update the shape property.
        N = x.shape[0]
        # Safety check for empty shapes
        if len(x.shape) > 1:
            total_features = 1
            for dim in x.shape[1:]:
                total_features *= dim
        else:
            total_features = x.shape[0]
            
        return Tensor(x.data, [N, total_features])

    def get_stats(self):
        return 0, 0, 0