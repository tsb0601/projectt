import copy
import torch
import torch.nn as nn

# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)
        self.register_buffer("buffer", torch.ones(1))
    def forward(self, x):
        return self.fc(x)

# Create the original model and its deep copy
original_model = SimpleModel()
copied_model = copy.deepcopy(original_model)

# Check if the weights are equal
weights_equal = torch.equal(original_model.fc.weight.data, copied_model.fc.weight.data)
buffer_equal = torch.equal(original_model.buffer.data, copied_model.buffer.data)
print("Are weights equal:", weights_equal)
print("Are buffers equal:", buffer_equal)

