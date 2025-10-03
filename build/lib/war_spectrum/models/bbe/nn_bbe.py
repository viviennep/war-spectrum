import numpy as np, polars as pl, pathlib, joblib, torch
from torch.utils.data.dataset import Dataset
cl = pl.col

class Whitener():
    def __init__(self):
        pass

    def fit(self, X, y=None):
        μ = X.mean(0)
        Xc = X - μ
        Σ = np.cov(Xc.T)
        L = np.linalg.cholesky(np.linalg.inv(Σ))
        self.L = L
        self.μ = μ
        return self

    def transform(self, X):
        Xc = X - self.μ
        return Xc@self.L

class MLPxwOBA(torch.nn.Module):
    def __init__(
        self, 
        in_dim: int,
        out_dim: int,
        hidden: int,
        depth: int,
        dropout: float,
        activation: str
    ):
        super().__init__()
        act = {
            'relu': torch.nn.ReLU,
            'gelu': torch.nn.GELU,
            'silu': torch.nn.SiLU,
            'tanh': torch.nn.Tanh
        }[activation]

        layers = []
        d = in_dim
        for _ in range(depth):
            layers += [torch.nn.Linear(d, hidden), act(), torch.nn.Dropout(dropout)]
            d = hidden
        layers += [torch.nn.Linear(d, out_dim)]
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def loss_fn(logits, y, α_brier=0.02, label_smoothing=0.02):
    # cross entropy with label smoothing + small brier regularizer
    ce = torch.nn.functional.cross_entropy(
        logits, y, label_smoothing=label_smoothing
    )
    if α_brier > 0:
        p = torch.softmax(logits, dim=1)
        onehot = torch.nn.functional.one_hot(y, num_classes=p.size(1)).float()
        brier = ((p - onehot)**2).sum(dim=1).mean()
        return ce + α_brier*brier
    return ce

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        logits = model(Xb)
        loss = loss_fn(logits, yb)
        loss_sum += loss.item()
        pred = logits.argmax(1)
        correct += (pred == yb).sum().item()
        total += yb.numel()
    return {'loss': loss_sum/total, 'acc': correct/total}

def train_model(
        model,
        train_loader,
        val_loader,
        device,
        epochs,
        lr,
        weight_decay,
        warmup_epochs=3,
    ):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # cosine schedule with warmup
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch+1)/max(1,warmup_epochs)
        t = (epoch-warmup_epochs)/max(1,(epochs-warmup_epochs))
        return 0.5*(1+np.cos(np.pi*t))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)

    best = {'val_loss': float('inf'), 'state': None, 'epoch': -1}
    patience, bad = 10, 0

    for ep in range(epochs):
        model.train()
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            logits = model(Xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        val = evaluate(model, val_loader, device)
        print(f"Epoch: {ep}, Val {val['loss']}")
        if val['loss'] < best['val_loss'] - 1e-5:
            best['val_loss'] = val['loss']
            best['state'] = {k:v.cpu() for k,v in model.state_dict().items()}
            best['epoch'] = ep
            bad = 0
        else:
            bad += 1
        if bad >= patience:
            break

    # load best
    if best['state'] is not None:
        model.load_state_dict(best['state'])
    return best

class DatasetFromNumpy(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return self.X[i], self.y[i]

features = ['launch_speed','launch_angle']
targets = ['is_1b','is_2b','is_3b','is_hr','is_sf','is_gidp','is_out']

def load_Xy(data_path):
    df = pl.read_parquet(data_path)
    X = df.filter('is_tracked').select(
        features, 
        cl('sprint_speed').fill_null(cl('sprint_speed').mean())
    ).to_numpy()
    y_raw = df.filter('is_tracked').select(targets).fill_null(False).to_numpy()
    y = y_raw.argmax(1)
    return X, y


