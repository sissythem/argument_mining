import torch
import torch.nn.functional as f


class FeedForward(torch.nn.Module):
    properties = None
    app_logger = None
    num_labels = 0

    def __init__(self, properties, logger, device_name, num_labels, input_size=768):
        super(FeedForward, self).__init__()
        self.properties = properties
        self.app_logger = logger
        self.device_name = device_name
        self.input_size = input_size
        self.ff = torch.nn.Linear(self.input_size, num_labels).to(device_name)

    def forward(self, x):
        hidden = self.ff(x)
        activation_function = self._get_activation_function()
        res = activation_function(hidden)
        output = f.dropout(res)
        output = f.softmax(output, dim=2)
        return output

    def _get_activation_function(self):
        activation_func_name = self.properties["model"]["activation_function"]
        if activation_func_name == "relu":
            act_function = f.relu
        elif activation_func_name == "sigmoid":
            act_function = torch.sigmoid
        elif activation_func_name == "tanh":
            act_function = torch.tanh
        elif activation_func_name == "leaky_relu":
            act_function = f.leaky_relu
        else:
            # ReLU activations by default
            act_function = f.relu
        return act_function
