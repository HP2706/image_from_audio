from common import stub, TRAIN_DIR_VOLUME
from modal import Image, gpu
from model import image



with image.imports():
    from torch import nn
    from torch.optim import Adam
    import torch.nn.functional as F

class Adapter(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers) -> None:
        super().__init__()
        self.l1 = nn.Linear(input_dim, hidden_dim)
        self.intermediate_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers)])
        self.non_lins = nn.ModuleList([nn.GELU() for _ in range(n_layers)])
        self.l_out = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        #x is of shape [batch_size, input_dim]
        x = self.l1(x)
        for intermediate_layer, non_lin in zip(self.intermediate_layers, self.non_lins):
            x = non_lin(intermediate_layer(x))
        return self.l_out(x)

@stub.function(    
    image = image,
    gpu = gpu.A10G()
)
def train():

    #define model
    model = Adapter(input_dim=1024, hidden_dim=1024, output_dim=1024, n_layers=2)
    n_epochs = 10
    lr = 0.001
    optimizer = Adam(model.parameters(), lr=lr)

    #load dataset
        
    for _ in range(n_epochs):
        for batch in dataloader:
            x, y = batch # y is of shape [batch_size, input_dim], x is of shape [batch_size, target_dim]
            y_pred = model(x)
            loss = F.mse_loss(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()



