

class TwoLayerNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, std=1e-4):
        super(TwoLayerNet, self).__init__()
        self.layer_1 = nn.Linear(input_dim, hidden_dim) .to(device)
        self.layer_out = nn.Linear(hidden_dim, output_dim).to(device)
        self.relu = nn.ReLU().to(device)

    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.layer_out(x)    
        return x