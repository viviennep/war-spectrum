import numpy as np, polars as pl, joblib
from sklearn.neighbors import KNeighborsClassifier
from catboost import CatBoostClassifier
from war_spectrum.models.utils import split_file,assemble_to_tempfile
cl = pl.col

class PitchQualityModel():
    def __init__(
        self,
        primary_features,
        secondary_features,
        pitch_types,
        submodel_masks,
        submodel_targets,
        submodel_catboost_params,
    ):
        '''
        Parameters:
        : primary_features & secondary_features: 
            lists of features for primary/secondary models to train with
            different for stuff vs. pitching
        : pitch_types
        : submodel_masks
            dictionary whose values contain polars expressions for isolating
            the appropriate rows via a `filter` call
            keys will be:
                'is_swing',
                'outcome_given_take',
                'outcome_given_swing',
                'outcome_given_bbe'
        : submodel_targets
            dictionary whose values contain column ids for different outcome classes
            keys that will be used are:
                'is_swing': binary classification
                'outcome_given_take': multiclassification (called strike/ball or hbp)
                'outcome_given_swing': multiclassification (whiff, foul, bbe)
                'outcome_given_bbe': multiclassification (1b,2b,3b,hr,sf,gidp,out)
        : submodel_catboost_params 
            dictionary of dictionaries of parameters for catboost models
            one dictionary per submodel
        '''
        self.primary_features         = primary_features
        self.secondary_features       = secondary_features
        self.pitch_types              = pitch_types
        self.submodel_masks           = submodel_masks
        self.submodel_targets         = submodel_targets
        self.submodel_catboost_params = submodel_catboost_params
        # Four sets of sub-models
        self.swing_models         = {}
        self.take_outcome_models  = {}
        self.swing_outcome_models = {}
        self.bbe_outcome_models   = {}
        # Keep ordering consistend
        self.class_order = {}
        self.out_order = [ 
            'p_ball',
            'p_called_strike',
            'p_hbp',
            'p_whiff',
            'p_foul',
            'p_1b',
            'p_2b',
            'p_3b',
            'p_hr',
            'p_sf',
            'p_gidp',
            'p_out',
        ]

    @staticmethod
    def create_multi_y(selected_df):
        return selected_df.to_numpy().argmax(1).astype(int)

    def make_train_test_split(self,df,train_inds=None,valid_inds=None):
        df = df.with_row_index()
        train_inds = train_inds if train_inds is not None else np.arange(len(df))
        valid_inds = np.delete(np.arange(len(df)),train_inds)
        tr_df = df.filter(cl('index').is_in(train_inds))
        va_df = df.filter(cl('index').is_in(valid_inds))
        return tr_df,va_df

    def make_eval_set(self, mask, va_df, features, targets, multi=False):
        va_X = va_df.filter(mask).select(features).to_numpy()
        va_y = va_df.filter(mask).select(targets)
        va_y = va_y.to_numpy().ravel() if not multi else self.create_multi_y(va_y)
        return (va_X, va_y)

    def fit_submodel(
            self,
            key, pitch_type,
            tr_df, va_df = None,
        ):
        mask = cl('pitch_type_classification').eq(pitch_type) \
               & self.submodel_masks[key]
        if 'primary' in pitch_type:
            features = self.primary_features
        else:
            features = self.secondary_features
        targets = self.submodel_targets[key]
        X = tr_df.filter(mask).select(features).to_numpy()
        if isinstance(targets,list):
            y = self.create_multi_y(tr_df.filter(mask).select(targets))
        else:
            y = tr_df.filter(mask).select(targets).to_numpy().ravel()
        submodel = CatBoostClassifier(**self.submodel_catboost_params[key])
        submodel.fit(
            X, y, 
            eval_set = None if va_df is None 
                            else self.make_eval_set(mask,va_df,features,targets,
                                                    multi=isinstance(targets,list))
        )
        return submodel

    def fit(
            self, 
            df: pl.DataFrame,
            train_inds = None,
            valid_inds = None
        ):

        tr_df,va_df = self.make_train_test_split(df,train_inds,valid_inds)

        for pitch_type in self.pitch_types:
            # Swing vs. Take 
            print(pitch_type, 'swing')
            key = 'is_swing'
            swing_model = self.fit_submodel(key, pitch_type, tr_df, va_df)
            self.swing_models[pitch_type] = swing_model
            self.class_order[(pitch_type,key)] = list(self.submodel_targets[key])

            # Outcome given take
            print(pitch_type, 'take outcome')
            key = 'outcome_given_take'
            take_model = self.fit_submodel(key, pitch_type, tr_df, va_df)
            self.take_outcome_models[pitch_type] = take_model
            self.class_order[(pitch_type,key)] = list(self.submodel_targets[key])

            # Outcome given swing
            print(pitch_type, 'swing outcome')
            key = 'outcome_given_swing'
            swing_outcome_model = self.fit_submodel(key, pitch_type, tr_df, va_df)
            self.swing_outcome_models[pitch_type] = swing_outcome_model
            self.class_order[(pitch_type,key)] = list(self.submodel_targets[key])

            # Outcome given BBE
            print(pitch_type, 'bbe outcome')
            key = 'outcome_given_bbe'
            bbe_outcome_model = self.fit_submodel(key, pitch_type, tr_df, va_df)
            self.bbe_outcome_models[pitch_type] = bbe_outcome_model
            self.class_order[(pitch_type,key)] = list(self.submodel_targets[key])

    def predict_proba(self, df: pl.DataFrame):
        '''
        Return array of unconditional probabilities for each outcome:
            ball          = p(ball    |  ¬swing)×(1-p(swing))
            called strike = p(strike  |  ¬swing)×(1-p(swing))
            hbp           = p(hbp     |  ¬swing)×(1-p(swing))
            whiff         = p(whiff   |   swing)×p(swing)
            foul          = p(foul    |   swing)×p(swing)
            1b            = p(single  | in play)×p(in play | swing)×p(swing)
            2b            = p(double  | in play)×p(in play | swing)×p(swing)
            3b            = p(triple  | in play)×p(in play | swing)×p(swing)
            hr            = p(homer   | in play)×p(in play | swing)×p(swing)
            sf            = p(sac fly | in play)×p(in play | swing)×p(swing)
            gidp          = p(gidp    | in play)×p(in play | swing)×p(swing)
            out           = p(out     | in play)×p(in play | swing)×p(swing)
        '''
        # Create output array 
        output_labels = [
            'ball','called_strike','hbp',         # take outcomes
            'whiff','foul',                       # swing outcomes
            '1b','2b','3b','hr','sf','gidp','out' # bip outcomes
        ]
        out = np.zeros((len(df),len(output_labels))) - 11111 # impossible value bug trap 
        for pitch_type in self.pitch_types:
            if 'primary' in pitch_type:
                features = self.primary_features
            else:
                features = self.secondary_features
            mask = cl('pitch_type_classification').eq(pitch_type)
            pitch_class_mask = df.select(mask).to_numpy().ravel()
            pitch_class_inds = np.where(pitch_class_mask)[0]
            if pitch_class_inds.size==0:
                continue

            X_pitch_type = df.filter(mask).select(features).to_numpy()

            # p(swing)
            p_swing = self.swing_models[pitch_type].predict_proba(X_pitch_type)[:,1]

            # p(ball,strike,hbp | ¬swing)
            key = 'outcome_given_take'
            p_take_outcomes = (
                self.take_outcome_models[pitch_type].predict_proba(X_pitch_type)
            )
            order = self.class_order[(pitch_type,key)]
            p_ball_given_take = p_take_outcomes[:,order.index('is_ball')]
            p_cs_given_take   = p_take_outcomes[:,order.index('is_called_strike')]
            p_hbp_given_take  = p_take_outcomes[:,order.index('is_hbp')]

            # p(whiff,foul,in play | swing)
            key = 'outcome_given_swing'
            p_swing_outcomes = (
                self.swing_outcome_models[pitch_type].predict_proba(X_pitch_type)
            )
            order = self.class_order[(pitch_type,key)]
            p_whiff_given_swing = p_swing_outcomes[:,order.index('is_whiff')]
            p_foul_given_swing  = p_swing_outcomes[:,order.index('is_foul')]
            p_bbe_given_swing   = p_swing_outcomes[:,order.index('is_bbe')]

            # p(1b,2b,3b,hr,sf,gidp,out | in play)
            key = 'outcome_given_bbe'
            p_bbe_outcomes = (
                self.bbe_outcome_models[pitch_type].predict_proba(X_pitch_type)
            )
            order = self.class_order[(pitch_type,key)]
            p_1b_given_bbe   = p_bbe_outcomes[:,order.index('is_1b')]
            p_2b_given_bbe   = p_bbe_outcomes[:,order.index('is_2b')]
            p_3b_given_bbe   = p_bbe_outcomes[:,order.index('is_3b')]
            p_hr_given_bbe   = p_bbe_outcomes[:,order.index('is_hr')]
            p_sf_given_bbe   = p_bbe_outcomes[:,order.index('is_sf')]
            p_gidp_given_bbe = p_bbe_outcomes[:,order.index('is_gidp')]
            p_out_given_bbe  = p_bbe_outcomes[:,order.index('is_out')]

            # Unconditionalize  
            p_take          = 1 - p_swing
            p_ball          = p_ball_given_take*p_take
            p_called_strike = p_cs_given_take*p_take
            p_hbp           = p_hbp_given_take*p_take
            p_whiff         = p_whiff_given_swing*p_swing
            p_foul          = p_foul_given_swing*p_swing
            p_bbe           = p_bbe_given_swing*p_swing
            p_1b            = p_1b_given_bbe*p_bbe
            p_2b            = p_2b_given_bbe*p_bbe
            p_3b            = p_3b_given_bbe*p_bbe
            p_hr            = p_hr_given_bbe*p_bbe
            p_sf            = p_sf_given_bbe*p_bbe
            p_gidp          = p_gidp_given_bbe*p_bbe
            p_out           = p_out_given_bbe*p_bbe

            out[pitch_class_inds] = np.c_[
                p_ball,
                p_called_strike,
                p_hbp,
                p_whiff,
                p_foul,
                p_1b,
                p_2b,
                p_3b,
                p_hr,
                p_sf,
                p_gidp,
                p_out,
            ]

        return out

    def save(self, path, split):
        out = {
            'init_params': {
                'pitch_types': self.pitch_types,
                'primary_features': self.primary_features,
                'secondary_features': self.secondary_features,
                #'submodel_masks': self.submodel_masks,
                'submodel_targets': self.submodel_targets,
                'submodel_catboost_params': self.submodel_catboost_params,
            },
            'submodels': {
                'swing_models': self.swing_models,
                'take_outcome_models': self.take_outcome_models,
                'swing_outcome_models': self.swing_outcome_models,
                'bbe_outcome_models': self.bbe_outcome_models,
            },
            'class_order': self.class_order,
            'out_order': self.out_order,
        }
        joblib.dump(out, path)
        if split:
            split_file(path)

    @classmethod
    def load(cls, path, split=True):
        if split:
            tmp = assemble_to_tempfile(path)
            info = joblib.load(tmp)
        else:
            info = joblib.load(path)
        model = cls(submodel_masks={}, **info['init_params'])
        model.class_order = info['class_order']
        model.swing_models = info['submodels']['swing_models']
        model.take_outcome_models = info['submodels']['take_outcome_models']
        model.swing_outcome_models = info['submodels']['swing_outcome_models']
        model.bbe_outcome_models = info['submodels']['bbe_outcome_models']
        model.out_order = info['out_order']
        return model


