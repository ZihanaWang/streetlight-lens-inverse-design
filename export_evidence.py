# run after pipeline_v2.py. writes the row-level files behind the summaries
# (inverse_surrogate_predictions.csv, inverse_lookup.csv, split_assignments.csv, seeds.json)
# and checks them against inverse.json
import json, os, sys
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
from common import P, M, PS, MS, load, composite_top, metrics

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get('RESULTS_DIR', os.path.join(HERE, 'results_v2'))
J = lambda n: json.load(open(os.path.join(OUT, n)))


def gpr(d, restarts=2):   # same as in pipeline_v2
    k = C(1.0) * Matern(length_scale=[1.0] * d, nu=2.5, length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(1e-2, (1e-6, 1e1))
    return MultiOutputRegressor(make_pipeline(StandardScaler(), GaussianProcessRegressor(k, normalize_y=True, n_restarts_optimizer=restarts, random_state=0)))


full, ef = load(1), load(2)
top, score = composite_top(ef, 30); rest = ef.drop(top)
jtr, jte = train_test_split(rest.index.values, test_size=0.2, random_state=42)
inv = J('inverse.json')
assert list(jtr) == inv['train_ids'] and list(jte) == inv['test_ids'] and list(map(int, top)) == inv['heldout_ids'], 'inverse split differs'
Ftr, Ttr = rest.loc[jtr, M].values, rest.loc[jtr, P].values
mean_T = Ttr.mean(0)
excl = set(jte) | set(top)
keep_ids = [i for i in full.index if i not in excl]
ev = gpr(4).fit(full.loc[keep_ids, P].values, full.loc[keep_ids, M].values)

rows, problems = [], []
mape = lambda y, p: np.mean(np.abs((y - p) / y), 0) * 100
for tag, ids in [('in', jte), ('ho', top)]:
    F, T = ef.loc[ids, M].values, ef.loc[ids, P].values
    true_pred, const_pred = ev.predict(T), ev.predict(np.tile(mean_T, (len(ids), 1)))
    member_mape = []
    for k in range(5):
        mem = pd.read_csv(os.path.join(OUT, f'inverse_{tag}_member{k}.csv'), index_col='Exp')
        assert list(mem.index) == list(ids)
        G = mem[[f'pred_{s}' for s in PS]].values
        Pk = ev.predict(G); member_mape.append(mape(F, Pk))
        for i, e in enumerate(ids):
            rows.append(dict(set=tag, Exp=int(e), source=f'inverse training {k}', **{f'geom_{s}': G[i, j] for j, s in enumerate(PS)},
                             **{f'target_{s}': F[i, j] for j, s in enumerate(MS)}, **{f'surrogate_{s}': Pk[i, j] for j, s in enumerate(MS)}))
    for name, G, Pg in [('true geometry', T, true_pred), ('constant geometry (training mean)', np.tile(mean_T, (len(ids), 1)), const_pred)]:
        for i, e in enumerate(ids):
            rows.append(dict(set=tag, Exp=int(e), source=name, **{f'geom_{s}': G[i, j] for j, s in enumerate(PS)},
                             **{f'target_{s}': F[i, j] for j, s in enumerate(MS)}, **{f'surrogate_{s}': Pg[i, j] for j, s in enumerate(MS)}))
    # check against inverse.json
    checks = [(f'surrogate_perf_{tag}', np.mean(member_mape, 0)), (f'surrogate_error_at_true_geometry_{tag}', mape(F, true_pred)),
              (f'surrogate_perf_const_{tag}', mape(F, const_pred))]
    for key, val in checks:
        ref = np.array([inv[key][s]['MAPE'] for s in MS])
        if not np.allclose(val, ref, atol=1e-6): problems.append((key, val.round(4).tolist(), ref.round(4).tolist()))
pd.DataFrame(rows).to_csv(os.path.join(OUT, 'inverse_surrogate_predictions.csv'), index=False)

# nearest design lookup
sF = StandardScaler().fit(Ftr); nn = NearestNeighbors(n_neighbors=1).fit(sF.transform(Ftr))
lk = []
for tag, ids in [('in', jte), ('ho', top)]:
    F = ef.loc[ids, M].values
    nb = nn.kneighbors(sF.transform(F), return_distance=False)[:, 0]
    for i, e in enumerate(ids):
        lk.append(dict(set=tag, Exp=int(e), nearest_training_Exp=int(jtr[nb[i]]), **{f'target_{s}': F[i, j] for j, s in enumerate(MS)},
                       **{f'nearest_recorded_{s}': Ftr[nb[i], j] for j, s in enumerate(MS)}))
    ref = np.array([inv[f'lookup_recorded_{tag}'][s]['MAPE'] for s in MS])
    if not np.allclose(mape(F, Ftr[nb]), ref, atol=1e-6): problems.append((f'lookup_recorded_{tag}', mape(F, Ftr[nb]).tolist(), ref.tolist()))
pd.DataFrame(lk).to_csv(os.path.join(OUT, 'inverse_lookup.csv'), index=False)

# splits
fw = J('forward_test.json'); folds = J('nested_cv_folds_ids.json')
fold_of = {e: d['fold'] for d in folds for e in d['test_ids']}
sa = pd.DataFrame(index=pd.Index(sorted(set(full.index) | set(ef.index)), name='Exp'))
sa['in_forward_dataset_496'] = sa.index.isin(full.index)
sa['forward_split'] = ['test' if e in set(fw['test_ids']) else 'train' if e in set(fw['train_ids']) else '' for e in sa.index]
sa['nested_cv_outer_test_fold'] = [fold_of.get(e, '') for e in sa.index]
sa['in_inverse_dataset_366'] = sa.index.isin(ef.index)
sa['inverse_split'] = ['heldout' if e in set(top) else 'test' if e in set(jte) else 'train' if e in set(jtr) else '' for e in sa.index]
sa['composite_score'] = score.reindex(sa.index)
sa.to_csv(os.path.join(OUT, 'split_assignments.csv'))
for name, n in [('forward train', (sa.forward_split == 'train').sum()), ('forward test', (sa.forward_split == 'test').sum()),
                ('inverse train', (sa.inverse_split == 'train').sum()), ('inverse test', (sa.inverse_split == 'test').sum()),
                ('inverse held-out', (sa.inverse_split == 'heldout').sum())]:
    print(f'{name}: {n}')

json.dump({
    'data_splits': {'forward train/test': 'train_test_split(CleanedData index, test_size=0.2, random_state=42)',
                    'nested CV outer folds': 'RepeatedKFold(n_splits=5, n_repeats=1, random_state=42)',
                    'inverse held-out': 'top 30 ErrorFree designs by composite score, no randomness',
                    'inverse train/test': 'train_test_split(remaining 336, test_size=0.2, random_state=42)',
                    'validation split inside every network': 'train_test_split(fitting rows, test_size=0.2, random_state=<network seed>)'},
    'network_seeds': {'network seed rule': 'fit_dnn(seed) gives member k the seed seed*10+k. seed_all seeds python, numpy and tensorflow',
                      'forward configuration selection': 'select_cfg seed 1000, inner KFold(3, shuffle, random_state=1000), inner fold j uses fit_dnn seed 1000+j',
                      'final forward model': 'fit_dnn seed 7 (members 70-74)',
                      'outer fold f, single network': 'fit_dnn seed 500+f',
                      'outer fold f, configuration selection': 'select_cfg seed 2000+10f',
                      'outer fold f, revised DNN': 'fit_dnn seed 600+f',
                      'inverse training k (k=0..4)': 'fit_member seed 100+k'},
    'other_seeds': {'Gaussian processes': 'random_state=0, 2 optimizer restarts', 'random forest, gradient boosting': 'random_state=0',
                    'Sobol samples and bootstrap': 'SALib sobol sample N=4096 seed=1, analyze seed=1',
                    'non-uniqueness search': 'scipy qmc.LatinHypercube(d=4, seed=3), 200,000 points'},
    'note': 'tensorflow is not bit-reproducible across machines, so the last digit can differ'},
    open(os.path.join(OUT, 'seeds.json'), 'w'), indent=1)

if problems:
    print('MISMATCH', problems); sys.exit(1)
print('ok, matches inverse.json')
