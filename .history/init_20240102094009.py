from sk

def init_model(X_train, y_train):
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
