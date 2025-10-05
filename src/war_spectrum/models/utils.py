import numpy as np, polars as pl, pathlib, tempfile, os
from catboost import CatBoostRegressor
cl = pl.col

def impute_arm_angle(df, model_dir):
    arm_angle_imputer = CatBoostRegressor().load_model(model_dir)
    X_impute_arm_angle = df.select(
        'release_pos_x',
        'release_pos_z',
        'release_extension',
        'height',
        'p_throws'
    ).to_numpy()
    pred_aa = arm_angle_imputer.predict(X_impute_arm_angle)
    df = df.with_columns(
        arm_angle = pl.when(cl('arm_angle').is_null()).then(pred_aa).otherwise('arm_angle')
    )
    return df

def split_file(src_path, split_mb=95):
    src = pathlib.Path(src_path)
    if not src.is_file():
        raise FileNotFoundError(src)

    size = src.stat().st_size
    split_size = split_mb*1024*1024
    n_parts = max(1,int(np.ceil(size/split_size)))
    width = max(3,len(str(n_parts - 1)))

    parts = []
    with src.open('rb') as f:
        for i in range(n_parts):
            out = src.with_suffix(src.suffix + f".{i:0{width}d}")
            with out.open('wb') as g:
                to_write = min(split_size, size-i*split_size)
                remaining = to_write
                while remaining:
                    b = f.read(min(2**20, remaining))
                    if not b:
                        break
                    g.write(b)
                    remaining -= len(b)
            parts.append(out)
    return parts

def assemble_to_file(prefix, out_path):
    prefix = pathlib.Path(prefix)
    out_path = pathlib.Path(out_path)
    base = prefix
    if prefix.suffixes and prefix.suffixes[-1].lstrip('.').isdigit():
        base = pathlib.Path(str(prefix)[: -len(prefix.suffixes[-1])])

    parts = sorted(
        [p for p in base.parent.glob(base.name + ".*") if p.suffix.lstrip(".").isdigit()],
        key=lambda p: int(p.suffix.lstrip(".")),
    )

    if not parts:
        raise FileNotFoundError(f"No parts found for prefix {base}")

    with out_path.open('wb') as out:
        for p in parts:
            with p.open('rb') as f:
                while True:
                    b = f.read(2**20)
                    if not b:
                        break
                    out.write(b)
    return out_path

def assemble_to_tempfile(prefix, suffix=None):
    prefix = pathlib.Path(prefix)
    base = prefix
    if prefix.suffixes and prefix.suffixes[-1].lstrip(".").isdigit():
        base = pathlib.Path(str(prefix)[: -len(prefix.suffixes[-1])])

    if suffix is None:
        suffix = base.suffix or ""

    fd, tmp = tempfile.mkstemp(prefix="_assembled_"+base.stem, suffix=suffix)
    os.close(fd)
    return assemble_to_file(base, tmp)


