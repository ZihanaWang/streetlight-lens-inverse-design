import os, random
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np, pandas as pd

P = ['Ellipse Radius (R1)', 'Ellipse Radius (R2)', 'Height (H)', 'Spline Angle']
M = ['Average Luminance', 'Overall Uniformity of Luminance', 'Longitudinal Uniformity of Luminance']
PS = ['R1', 'R2', 'H', 'theta']
MS = ['L_ave', 'U0', 'U_l']
BOUNDS = np.array([[9.7, 10.7], [6, 20], [6, 7], [50, 63]], float)
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data.xlsx')


THRESHOLDS = {'Average Luminance': 0.75, 'Overall Uniformity of Luminance': 0.4, 'Longitudinal Uniformity of Luminance': 0.6}


# sheet 1 = CleanedData (496 designs, forward model)
# sheet 2 = ErrorFree (369 passed the merit function). keep the 366 that meet all 3 thresholds, drops 69, 210, 280
def load(sheet):
    d = pd.read_excel(DATA, sheet_name=sheet).set_index('Exp')
    if sheet == 2:
        ok = np.ones(len(d), bool)
        for col, lo in THRESHOLDS.items(): ok &= d[col].values >= lo
        d = d[ok]
    return d


def seed_all(s):
    import tensorflow as tf
    random.seed(s); np.random.seed(s); tf.random.set_seed(s)


def mlp(n_in, n_out, dropout=0.0):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Input, Dropout
    layers = [Input(shape=(n_in,)), Dense(64, activation='relu')]
    if dropout: layers.append(Dropout(dropout))
    layers.append(Dense(32, activation='relu'))
    if dropout: layers.append(Dropout(dropout))
    layers += [Dense(16, activation='relu'), Dense(n_out)]
    return Sequential(layers)


def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    mse = np.mean((y - p) ** 2, 0)
    mae = np.mean(np.abs(y - p), 0)
    mape = np.mean(np.abs((y - p) / y), 0) * 100
    r2 = 1 - np.sum((y - p) ** 2, 0) / np.sum((y - y.mean(0)) ** 2, 0)
    return dict(MSE=mse, RMSE=np.sqrt(mse), MAE=mae, MAPE=mape, R2=r2)


def table(y, p, names):
    m = metrics(y, p)
    return pd.DataFrame({k: v for k, v in m.items()}, index=names)


# composite score = mean of the min-max normalized metrics (same as LensDesign2/3)
def composite_top(df, n=30):
    x = df[M].values
    z = (x - x.min(0)) / (x.max(0) - x.min(0))
    s = pd.Series(z.mean(1), index=df.index)
    return s.sort_values(ascending=False).index[:n], s
