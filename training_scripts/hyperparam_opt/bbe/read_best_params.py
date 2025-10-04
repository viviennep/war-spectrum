import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

storage = JournalStorage(JournalFileBackend(f"cb-bbe.study.journal"))
study = optuna.load_study(study_name=f"cb_ss_bbe", storage=storage)
t = study.best_trial

