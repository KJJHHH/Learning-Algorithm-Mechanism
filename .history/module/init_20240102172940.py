from sklearn.linear_model import LinearRegression
import torch
from module.model import TwoLayerNet

# 1. initializing_1_ReLU_LR | L11 p2
# Get weight by ols and set weight of two layers by the weight
def init_model(X_train, y_train, dtype = torch.float64):
    miny = min(y_train)
    model = LinearRegression()
    model.fit(X_train, (y_train - miny))
    w = torch.tensor(model.coef_, dtype=torch.float32).reshape(1, -1)
    b = torch.tensor(model.intercept_, dtype=torch.float32).reshape(1)

    model = TwoLayerNet(X_train.shape[1], 1, 1)
    model.layer_1.weight.data = w
    model.layer_1.bias.data = b
    model.layer_out.weight.data = torch.tensor(1, dtype=torch.float32).reshape(1, 1)
    model.layer_out.bias.data = miny.reshape(1)
    return model
