import torch, json, pathlib
from torch import nn

class CatcherAndPitcherREs(nn.Module):
    def __init__(
            self,
            prior_σ_p=1.,
            prior_σ_c=1.,
            catcher_weights=None,
            pitcher_weights=None
        ):
        super().__init__()
        self.catcher_res = nn.ParameterDict()  
        self.pitcher_res = nn.ParameterDict()  
        self.prior_τ_p = 1./prior_σ_p**2
        self.prior_τ_c = 1./prior_σ_c**2
        self.intercept = nn.Parameter(torch.zeros(()))
        self.β = nn.Parameter(torch.ones(1))
        self.weights_c = {str(k): float(v) for k,v in (catcher_weights or {}).items()}
        self.weights_p = {str(k): float(v) for k,v in (pitcher_weights or {}).items()}

    def get_re(self, key, mode):
        re_dict = self.pitcher_res if mode=='pitcher' else self.catcher_res
        if key not in re_dict:
            re_dict[key] = nn.Parameter(torch.zeros(()))
        return re_dict[key]

    def center(self, d, weights_map):
        keys = list(d.keys())
        params = torch.stack([d[k] for k in keys])
        w = torch.tensor(
            [weights_map.get(k, 1.0) for k in keys],
            dtype=params.dtype,
            device=params.device
        )
        mean = (w*params).sum()/w.sum()
        return {k: (p - mean) for k, p in zip(keys, params)}, mean

    def forward(self, pred_logit, catchers, pitchers):
        pred_logit = torch.as_tensor(pred_logit)
        c_centered, c_mean = self.center(self.catcher_res, self.weights_c)
        p_centered, p_mean = self.center(self.pitcher_res, self.weights_p)
        re_c = torch.stack([c_centered[k] for k in catchers])
        re_p = torch.stack([p_centered[k] for k in pitchers])
        η = self.β*pred_logit + re_c + re_p + self.intercept
        return η, re_c, re_p

    def nll(self, y, η):
        return nn.functional.binary_cross_entropy_with_logits(η,y)

    def penalty(self):
        c_centered, c_mean = self.center(self.catcher_res, self.weights_c)
        p_centered, p_mean = self.center(self.pitcher_res, self.weights_p)
        catchers = sum(p.pow(2) for p in c_centered.values())/len(c_centered)
        pitchers = sum(p.pow(2) for p in p_centered.values())/len(p_centered)
        return 0.5*self.prior_τ_p*pitchers + 0.5*self.prior_τ_c*catchers

def initialize_new_player_params(model, catchers, pitchers):
    with torch.no_grad():
        for k in catchers: _ = model.get_re(k, 'catcher')
        for k in pitchers: _ = model.get_re(k, 'pitcher')

def train(model, base_logit, y, catchers, pitchers, max_iter=100):
    base_logit = torch.as_tensor(base_logit).float()
    y = torch.as_tensor(y).float()
    opt = torch.optim.LBFGS(
        model.parameters(), 
        lr=1.,
        line_search_fn='strong_wolfe',
        max_iter=max_iter,
    )
    lbfgs_calls = 0
    def closure():
        nonlocal lbfgs_calls
        opt.zero_grad()
        η, _, _ = model(base_logit, catchers, pitchers)
        loss = model.nll(y, η) + model.penalty()
        loss.backward()
        print(lbfgs_calls,loss.item())
        lbfgs_calls += 1
        return loss
    for _ in range(2):
        loss = opt.step(closure)
    return float(loss)

def save_model(model, model_dir, model_name='framing_random_effects'):
    path = pathlib.Path(model_dir).resolve()
    path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path / f'{model_name}.pt')
    meta = {
        'catcher_keys': list(model.catcher_res.keys()),
        'pitcher_keys': list(model.pitcher_res.keys()),
        'prior_τ_p': model.prior_τ_p,
        'prior_τ_c': model.prior_τ_c,
        'catcher_weights': model.weights_c,
        'pitcher_weights': model.weights_p,
    }
    (path/f'{model_name}.json').write_text(json.dumps(meta))

def load_model(model_dir, model_name='framing_random_effects', device='cpu'):
    path = pathlib.Path(model_dir).resolve()
    meta = json.loads((path/f'{model_name}.json').read_text())
    m = CatcherAndPitcherREs(
        prior_σ_p=(1./meta['prior_τ_p'])**0.5,
        prior_σ_c=(1./meta['prior_τ_c'])**0.5,
        catcher_weights = meta['catcher_weights'],
        pitcher_weights = meta['pitcher_weights'],
    ).to(device)
    with torch.no_grad():
        for k in meta['catcher_keys']: _ = m.get_re(k, 'catcher')
        for k in meta['pitcher_keys']: _ = m.get_re(k, 'pitcher')
    state = torch.load(
        path/f'{model_name}.pt', 
        map_location=device,
        weights_only=True
    )
    m.load_state_dict(state)
    return m


