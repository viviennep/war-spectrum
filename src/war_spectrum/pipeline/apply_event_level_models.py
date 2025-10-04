import numpy as np, polars as pl, pathlib
cl = pl.col

#def apply_bbe_classifier(df, model_path):
#    from war_spectrum.models.bbe import load_model, features, targets
#    bbe_classifier = load_model(model_path)
#    X = df.filter('is_tracked').select(features).to_numpy()
#    pred_y = bbe_classifier.predict_proba(X)
#    track_mask = df.select('is_tracked').to_numpy().squeeze()
#    y = np.zeros((len(df),pred_y.shape[1]))
#    y[track_mask] = pred_y
#    new_labels = [f"pred_{i.split('_')[1]}" for i in targets]
#    bbe_df = pl.DataFrame(dict(zip(new_labels,y.T)))
#    return df.hstack(bbe_df)

def apply_bbe_classifier(df, model_path):
    from war_spectrum.models.bbe.cb_bbe import load_model, features, targets
    bbe_classifier = load_model(model_path)
    X = df.filter('is_tracked').select(features).to_numpy()
    pred_y = bbe_classifier.predict_proba(X)
    track_mask = df.select('is_tracked').to_numpy().squeeze()
    y = np.zeros((len(df),pred_y.shape[1]))
    y[track_mask] = pred_y
    new_labels = [f"pred_{i.split('_')[1]}" for i in targets]
    bbe_df = pl.DataFrame(dict(zip(new_labels,y.T)))
    return df.hstack(bbe_df)

def apply_stuff(df, model_path):
    from war_spectrum.models.pitch_quality import PitchQualityModel
    model_path = pathlib.Path(model_path).resolve()
    stuff_model = PitchQualityModel.load(model_path)
    pred_stuff = stuff_model.predict_proba(df)
    new_labels = [f"{i}_stuff" for i in stuff_model.out_order]
    stuff_df = pl.DataFrame(dict(zip(new_labels,pred_stuff.T)))
    return df.hstack(stuff_df)

def apply_pitching(df, model_path):
    from war_spectrum.models.pitch_quality import PitchQualityModel
    model_path = pathlib.Path(model_path).resolve()
    pitch_model = PitchQualityModel.load(model_path)
    pred_pitch = pitch_model.predict_proba(df)
    new_labels = [f"{i}_pitch" for i in pitch_model.out_order]
    pitch_df = pl.DataFrame(dict(zip(new_labels,pred_pitch.T)))
    return df.hstack(pitch_df)

def apply_framing(df, model_dir, years=[2025]):
    from war_spectrum.models.framing import (
        train_framing_model,
        predict_called_strike_prob
    )
    model_dir = pathlib.Path(model_dir)
    # Yeah I re-train the framing model each time
    reduced_df = df.filter(cl('season').is_in(years))
    train_framing_model(
        reduced_df,
        model_dir = model_dir,
        model_name = 'framing_random_effects',
        train_base_model=False,
        test_train_split=0.9,
        pitcher_prior_σ = 1.,
        catcher_prior_σ = 5e1,
        warm_start=True,
    )
    pred_strike, avg_catcher =  predict_called_strike_prob(df, model_dir)
    return df.with_columns(
        p_called_strike_catcher = pred_strike,
        p_called_strike_avg = avg_catcher,
    )


