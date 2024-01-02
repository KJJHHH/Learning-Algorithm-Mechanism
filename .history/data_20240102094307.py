
class MyDataset(Data.Dataset):
    def __init__(self, X, y):

        self.X = X
        self.y = y

    def __getitem__(self, index):
        X_, y_ = self.X[index], self.y[index]
        return X_, y_

    def __len__(self):
        return len(self.X)