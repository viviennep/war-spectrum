import optuna, json
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

def load_best_params(model,submodel):
    storage = JournalStorage(JournalFileBackend(f"{model}.{submodel}.journal"))
    study = optuna.load_study(study_name=f"{model}-{submodel}", storage=storage)
    t = study.best_trial
    return dict(t.params)

model = 'pitch'
submodels = ['is_swing','outcome_given_take','outcome_given_swing','outcome_given_bbe']

params = {}
for submodel in submodels:
    best = load_best_params(model,submodel)
    params[submodel] = {k.split('.')[-1]: v for k,v in best.items()}

with open(f'{model}_params.json','w') as f:
    json.dump(params,f)

