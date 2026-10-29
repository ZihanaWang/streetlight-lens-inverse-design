# python pipeline_v2.py  (takes ~3 h on 2 cores)
# reads data.xlsx (= Oct_15_Lens_StructuralParameter_Ranges.xlsx), writes everything to results_v2/
import os, sys, json, time
from common import *
from sklearn.model_selection import train_test_split, KFold, RepeatedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
from scipy.stats import qmc
import warnings; warnings.filterwarnings('ignore')
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get('RESULTS_DIR', os.path.join(HERE, 'results_v2')); os.makedirs(OUT, exist_ok=True)
T0 = time.time()
def log(*a): print(f'[{(time.time()-T0)/60:6.1f} min]', *a, flush=True)
def save(name, obj): json.dump(obj, open(os.path.join(OUT, name), 'w'), indent=1, default=float)

# models
def gpr(d, restarts=2):
    k = C(1.0) * Matern(length_scale=[1.0] * d, nu=2.5, length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(1e-2, (1e-6, 1e1))
    return MultiOutputRegressor(make_pipeline(StandardScaler(), GaussianProcessRegressor(k, normalize_y=True, n_restarts_optimizer=restarts, random_state=0)))

def gp_residual_sd(m):
    return np.array([np.sqrt(e.steps[-1][1].kernel_.k2.noise_level) * e.steps[-1][1]._y_train_std for e in m.estimators_])

BASELINES = {
    'Linear regression': lambda: make_pipeline(StandardScaler(), LinearRegression()),
    'Quadratic regression': lambda: make_pipeline(StandardScaler(), PolynomialFeatures(2), RidgeCV(np.logspace(-4, 3, 30))),
    'Cubic regression': lambda: make_pipeline(StandardScaler(), PolynomialFeatures(3), RidgeCV(np.logspace(-4, 3, 30))),
    'k-nearest neighbours': lambda: GridSearchCV(make_pipeline(StandardScaler(), KNeighborsRegressor(weights='distance')),
                                                 {'kneighborsregressor__n_neighbors': [3, 5, 8, 12, 20]}, cv=5),
    'Support vector regression': lambda: MultiOutputRegressor(GridSearchCV(make_pipeline(StandardScaler(), SVR()),
                                                 {'svr__C': [0.1, 1, 10], 'svr__gamma': ['scale', 0.1, 0.3], 'svr__epsilon': [0.005, 0.01, 0.02]}, cv=5)),
    'Random forest': lambda: RandomForestRegressor(500, min_samples_leaf=2, random_state=0, n_jobs=-1),
    'Gradient boosting': lambda: MultiOutputRegressor(GradientBoostingRegressor(n_estimators=400, learning_rate=0.03, max_depth=3, subsample=0.8, random_state=0)),
    'Gaussian process': lambda: gpr(4),
}

# inverse net
INV_CFG = dict(widths=(64, 32, 16), act='relu', wd=0.0, lr=1e-3)
INV_DROPOUT, INV_TRAININGS, INV_SEED0 = 0.3, 5, 100   # training k uses seed INV_SEED0 + k
INVSEL_SEED, INVSEL_PATIENCES = 4000, (10, 50)
SPLIT_SEED = 42
DNN_CFG = {
    'A: 64-32-16 ReLU': dict(widths=(64, 32, 16), act='relu', wd=0.0, lr=1e-3, n=1),
    'B: 64-32-16 ReLU + L2': dict(widths=(64, 32, 16), act='relu', wd=1e-4, lr=1e-3, n=1),
    'C: ensemble of 5 x B': dict(widths=(64, 32, 16), act='relu', wd=1e-4, lr=1e-3, n=5),
}

def net(n_in, n_out, widths, act, wd, drop=0.0):
    L = [Input(shape=(n_in,))]
    for i, w in enumerate(widths):
        L.append(Dense(w, activation=act, kernel_regularizer=l2(wd) if wd else None))
        if drop and i < 2: L.append(Dropout(drop))
    L.append(Dense(n_out))
    return Sequential(L)

def fit_member(X, Y, cfg, seed, scaler=StandardScaler, drop=0.0, lr_pat=50):
    # 20% of X for validation, scalers fit on the other 80%
    a, v = train_test_split(np.arange(len(X)), test_size=0.2, random_state=seed)
    sx, sy = scaler().fit(X[a]), scaler().fit(Y[a])
    seed_all(seed)
    m = net(X.shape[1], Y.shape[1], cfg['widths'], cfg['act'], cfg['wd'], drop)
    m.compile(optimizer=Adam(cfg['lr']), loss='mse')
    h = m.fit(sx.transform(X[a]), sy.transform(Y[a]), validation_data=(sx.transform(X[v]), sy.transform(Y[v])),
              epochs=3000, batch_size=32, verbose=0,
              callbacks=[EarlyStopping(patience=150, restore_best_weights=True),
                         ReduceLROnPlateau(factor=0.5, patience=lr_pat, min_lr=1e-6)])
    return (lambda Z: sy.inverse_transform(m.predict(sx.transform(Z), verbose=0))), len(h.history['loss'])

def fit_dnn(X, Y, cfg, seed):
    members = [fit_member(X, Y, cfg, seed * 10 + k)[0] for k in range(cfg['n'])]
    return lambda Z: np.mean([f(Z) for f in members], 0)

def norm_mae(Y, P, ref_sd):
    return float(np.mean(np.mean(np.abs(Y - P), 0) / ref_sd))

INNER_FOLDS = {}   # filled by select_cfg

def select_cfg(X, Y, seed):
    # inner 3-fold CV, score = MAE / training SD, averaged over the outputs
    sd = Y.std(0); scores = {}
    for name, cfg in DNN_CFG.items():
        s = []
        for j, (a, b) in enumerate(KFold(3, shuffle=True, random_state=seed).split(X)):
            s.append(norm_mae(Y[b], fit_dnn(X[a], Y[a], cfg, seed + j)(X[b]), sd))
        scores[name] = float(np.mean(s)); INNER_FOLDS[name] = s
    return min(scores, key=scores.get), scores

def tab(Y, P, names):
    m = metrics(Y, P)
    return {n: {k: float(v[j]) for k, v in m.items()} for j, n in enumerate(names)}

full = load(1); ef = load(2)
# don't mix results from different data or settings
import hashlib
from common import THRESHOLDS
_fp = dict(data_sha256=hashlib.sha256(open(DATA, 'rb').read()).hexdigest(), thresholds=THRESHOLDS, bounds=BOUNDS.tolist(),
           dnn_cfg={k: {kk: (list(vv) if isinstance(vv, tuple) else vv) for kk, vv in v.items()} for k, v in DNN_CFG.items()},
           n_forward=len(full), n_inverse=len(ef), split_seed=SPLIT_SEED,
           inverse=dict(cfg={k: (list(v) if isinstance(v, tuple) else v) for k, v in INV_CFG.items()}, dropout=INV_DROPOUT,
                        trainings=INV_TRAININGS, seed0=INV_SEED0, patience_selection_seed=INVSEL_SEED, patience_candidates=list(INVSEL_PATIENCES)))
_fpf = os.path.join(OUT, 'run_fingerprint.json')
if os.path.exists(_fpf) and json.load(open(_fpf)) != json.loads(json.dumps(_fp)):
    sys.exit(f'{OUT} has results from other data/settings, use a new RESULTS_DIR')
json.dump(_fp, open(_fpf, 'w'), indent=1)
X, Y = full[P].values, full[M].values
gp_all = gpr(4).fit(X, Y); RES_SD = gp_residual_sd(gp_all)
save('dataset.json', dict(n_cleaned=len(full), n_errorfree=len(ef), gp_residual_sd=dict(zip(MS, RES_SD.tolist()))))
log('data loaded; GP residual SD', RES_SD.round(4))
# stages: fwd, cv [folds], cvmerge, invsel, inv, sens, reach, lt30. finished ones are skipped
ARGS = sys.argv[1:] or ['fwd', 'cv', 'cvmerge', 'invsel', 'inv', 'sens', 'reach', 'lt30']
FOLDS = [int(a) for a in ARGS if a.isdigit()] or list(range(5))
have = lambda n: os.path.exists(os.path.join(OUT, n))

# forward model
itr, ite = train_test_split(full.index.values, test_size=0.2, random_state=SPLIT_SEED)
if 'fwd' in ARGS and not have('forward_test.json'):
  Xtr, Ytr, Xte, Yte = full.loc[itr, P].values, full.loc[itr, M].values, full.loc[ite, P].values, full.loc[ite, M].values
  chosen, inner = select_cfg(Xtr, Ytr, 1000)
  log('forward selection on 396 training designs:', chosen, inner)
  pf = fit_dnn(Xtr, Ytr, DNN_CFG[chosen], 7)(Xte)
  gp_t = gpr(4).fit(Xtr, Ytr)
  save('forward_test.json', dict(chosen=chosen, inner_scores=inner, inner_fold_scores=dict(INNER_FOLDS), selection_seed=1000, final_seed=7, n_train=len(itr), n_test=len(ite),
                                 dnn=tab(Yte, pf, MS), gp=tab(Yte, gp_t.predict(Xte), MS), train_ids=itr.tolist(), test_ids=ite.tolist()))
  pd.DataFrame(np.column_stack([Yte, pf]), columns=[f'true_{s}' for s in MS] + [f'pred_{s}' for s in MS], index=pd.Index(ite, name='Exp')).to_csv(os.path.join(OUT, 'forward_test_predictions.csv'))
  log('forward test done', {s: round(v['MAPE'], 2) for s, v in tab(Yte, pf, MS).items()})

# nested 5-fold CV against the conventional models
outer = list(RepeatedKFold(n_splits=5, n_repeats=1, random_state=SPLIT_SEED).split(X))
for f, (tr, te) in enumerate(outer):
    if 'cv' not in ARGS or f not in FOLDS or have(f'cv_fold{f}.csv'): continue
    rows, preds = [], []
    keep = lambda name, Q: preds.extend(dict(fold=f, model=name, Exp=int(e), **{f'true_{o}': float(Y[te][i, j]) for j, o in enumerate(MS)},
                                             **{f'pred_{o}': float(Q[i, j]) for j, o in enumerate(MS)}) for i, e in enumerate(full.index.values[te]))
    for name, mk in BASELINES.items():
        Pr = mk().fit(X[tr], Y[tr]).predict(X[te]); keep(name, Pr)
        mm = metrics(Y[te], Pr)
        rows += [dict(fold=f, model=name, output=o, **{k: float(v[j]) for k, v in mm.items()}) for j, o in enumerate(MS)]
    Pr = fit_dnn(X[tr], Y[tr], dict(widths=(64, 32, 16), act='relu', wd=0.0, lr=1e-3, n=1), 500 + f)(X[te]); keep('DNN, first configuration', Pr)
    mm = metrics(Y[te], Pr)
    rows += [dict(fold=f, model='DNN, first configuration', output=o, **{k: float(v[j]) for k, v in mm.items()}) for j, o in enumerate(MS)]
    ch, sc = select_cfg(X[tr], Y[tr], 2000 + 10 * f)
    save(f'cv_fold{f}_inner_selection.json', dict(fold=f, seed=2000 + 10 * f, chosen=ch, mean_scores=sc, inner_fold_scores=dict(INNER_FOLDS),
                                                    score='mean over the outputs of MAE / training SD, inner 3-fold CV'))
    Pr = fit_dnn(X[tr], Y[tr], DNN_CFG[ch], 600 + f)(X[te]); keep('DNN, revised (nested selection)', Pr)
    mm = metrics(Y[te], Pr)
    rows += [dict(fold=f, model='DNN, revised (nested selection)', output=o, chosen=ch, **{k: float(v[j]) for k, v in mm.items()}) for j, o in enumerate(MS)]
    pd.DataFrame(rows).to_csv(os.path.join(OUT, f'cv_fold{f}.csv'), index=False)
    pd.DataFrame(preds).to_csv(os.path.join(OUT, f'cv_fold{f}_predictions.csv'), index=False)
    log(f'outer fold {f} done; inner choice {ch}')
if 'cvmerge' in ARGS and all(have(f'cv_fold{f}.csv') for f in range(5)):
  R = pd.concat([pd.read_csv(os.path.join(OUT, f'cv_fold{f}.csv')) for f in range(5)], ignore_index=True); R.to_csv(os.path.join(OUT, 'nested_cv_folds.csv'), index=False)
  per = R.groupby(['model', 'output'])[['MAE', 'R2']].agg(['mean', 'std'])
  avg = R.groupby(['model', 'fold'])[['MAE', 'R2']].mean().groupby('model').agg(['mean', 'std'])
  per.to_csv(os.path.join(OUT, 'nested_cv_per_output.csv')); avg.to_csv(os.path.join(OUT, 'nested_cv_average.csv'))
  save('nested_cv_folds_ids.json', [dict(fold=f, test_ids=full.index.values[te].tolist()) for f, (tr, te) in enumerate(outer)])
  log('nested CV done'); print(avg.round(4))

# inverse model
top, score = composite_top(ef, 30); rest = ef.drop(top)
jtr, jte = train_test_split(rest.index.values, test_size=0.2, random_state=SPLIT_SEED)
Ftr, Ttr = rest.loc[jtr, M].values, rest.loc[jtr, P].values
Fin, Tin = rest.loc[jte, M].values, rest.loc[jte, P].values
Fho, Tho = ef.loc[top, M].values, ef.loc[top, P].values
# lr patience for the inverse net (inner CV on the train designs)
if 'invsel' in ARGS and not have('inverse_patience_selection.json'):
  cfg_inv = INV_CFG
  sdT = Ttr.std(0); res = {}
  for pat in INVSEL_PATIENCES:
      s = []
      for j, (a, b) in enumerate(KFold(3, shuffle=True, random_state=INVSEL_SEED).split(Ftr)):
          for r in range(2):
              f_, _ = fit_member(Ftr[a], Ttr[a], cfg_inv, INVSEL_SEED + 10 * j + r, scaler=MinMaxScaler, drop=INV_DROPOUT, lr_pat=pat)
              s.append(norm_mae(Ttr[b], np.clip(f_(Ftr[b]), BOUNDS[:, 0], BOUNDS[:, 1]), sdT))
      res[str(pat)] = dict(mean=float(np.mean(s)), runs=[float(x) for x in s])
      log('inverse patience', pat, 'inner score', round(float(np.mean(s)), 4))
  save('inverse_patience_selection.json', dict(scores=res, chosen=int(min(res, key=lambda k: res[k]['mean'])),
       score='mean over the 4 parameters of MAE / training SD, inner 3-fold CV on the inverse train designs, 2 nets per fold'))

if 'inv' in ARGS and not have('inverse.json'):
  if not have('inverse_patience_selection.json'): sys.exit('run invsel first')
  INV_PAT = json.load(open(os.path.join(OUT, 'inverse_patience_selection.json')))['chosen']
  log('inverse lr patience', INV_PAT)
  cfg_inv = INV_CFG
  excl = set(jte) | set(top)
  ev = gpr(4).fit(full.drop(index=[i for i in excl if i in full.index])[P].values, full.drop(index=[i for i in excl if i in full.index])[M].values)
  members, clip = [], []
  for k in range(INV_TRAININGS):
      f_, ep = fit_member(Ftr, Ttr, cfg_inv, INV_SEED0 + k, scaler=MinMaxScaler, drop=INV_DROPOUT, lr_pat=INV_PAT)
      raw_in, raw_ho = f_(Fin), f_(Fho)
      clip.append(dict(member=k, epochs=ep,
                       in_frac_values_clipped=float(np.mean((raw_in < BOUNDS[:, 0]) | (raw_in > BOUNDS[:, 1]))),
                       ho_frac_values_clipped=float(np.mean((raw_ho < BOUNDS[:, 0]) | (raw_ho > BOUNDS[:, 1])))))
      p_in, p_ho = np.clip(raw_in, BOUNDS[:, 0], BOUNDS[:, 1]), np.clip(raw_ho, BOUNDS[:, 0], BOUNDS[:, 1])
      members.append((p_in, p_ho))
      for tag, ids, Tt, pp in [('in', jte, Tin, p_in), ('ho', top, Tho, p_ho)]:
          pd.DataFrame(np.column_stack([Tt, pp]), columns=[f'true_{s}' for s in PS] + [f'pred_{s}' for s in PS],
                       index=pd.Index(ids, name='Exp')).to_csv(os.path.join(OUT, f'inverse_{tag}_member{k}.csv'))
      log('inverse member', k, 'done')

  def mean_sd(tabs):
      return {s: {**{k: float(np.mean([t[s][k] for t in tabs])) for k in tabs[0][s]},
                  **{k + '_sd': float(np.std([t[s][k] for t in tabs], ddof=1)) for k in tabs[0][s]}} for s in tabs[0]}

  mean_T = Ttr.mean(0)
  nn = NearestNeighbors(n_neighbors=1).fit(StandardScaler().fit(Ftr).transform(Ftr)); sF = StandardScaler().fit(Ftr)
  inv = dict(n_train=len(jtr), n_test=len(jte), n_heldout=len(top), train_ids=jtr.tolist(), test_ids=jte.tolist(), heldout_ids=list(map(int, top)),
             clipping=clip,
             param_in=mean_sd([tab(Tin, m[0], PS) for m in members]), param_ho=mean_sd([tab(Tho, m[1], PS) for m in members]),
             const_param_in=tab(Tin, np.tile(mean_T, (len(Tin), 1)), PS), const_param_ho=tab(Tho, np.tile(mean_T, (len(Tho), 1)), PS))
  for tag, F, T, idx in [('in', Fin, Tin, 0), ('ho', Fho, Tho, 1)]:
      inv[f'surrogate_perf_{tag}'] = mean_sd([tab(F, ev.predict(m[idx]), MS) for m in members])
      inv[f'surrogate_error_at_true_geometry_{tag}'] = tab(F, ev.predict(T), MS)
      inv[f'surrogate_perf_const_{tag}'] = tab(F, ev.predict(np.tile(mean_T, (len(F), 1))), MS)
      nb = nn.kneighbors(sF.transform(F), return_distance=False)[:, 0]
      inv[f'lookup_recorded_{tag}'] = tab(F, Ftr[nb], MS)   # nearest training design
      inv[f'lookup_surrogate_{tag}'] = tab(F, ev.predict(Ttr[nb]), MS)
  inv['composite'] = dict(rest_max=float(score.drop(top).max()), held_min=float(score.loc[top].min()),
                          held_Ul_max=float(ef.loc[top, M[2]].max()), rest_Ul_max=float(rest[M[2]].max()))
  save('inverse.json', inv); log('inverse done')

# sobol indices and non-uniqueness, GP on all 496
top, score = composite_top(ef, 30); rest = ef.drop(top); jtr, jte = train_test_split(rest.index.values, test_size=0.2, random_state=SPLIT_SEED); Fin = rest.loc[jte, M].values
if 'sens' in ARGS and not have('sensitivity.json'):
  from SALib.sample import sobol as ssob
  from SALib.analyze import sobol as asob
  prob = {'num_vars': 4, 'names': PS, 'bounds': BOUNDS.tolist()}
  Xs = ssob.sample(prob, 4096, calc_second_order=False, seed=1); Ys = gp_all.predict(Xs)
  sob = {}
  for j, o in enumerate(MS):
      a = asob.analyze(prob, Ys[:, j], calc_second_order=False, seed=1)
      sob[o] = {p: dict(S1=float(a['S1'][i]), S1_conf=float(a['S1_conf'][i]), ST=float(a['ST'][i]), ST_conf=float(a['ST_conf'][i])) for i, p in enumerate(PS)}
  lhs = qmc.scale(qmc.LatinHypercube(d=4, seed=3).random(200000), BOUNDS[:, 0], BOUNDS[:, 1]); Yl = gp_all.predict(lhs)
  nu_rows = []
  for mult in (0.5, 1.0, 2.0):
      for eid, t in zip(jte, Fin):
          ok = np.all(np.abs(Yl - t) <= mult * RES_SD, axis=1); r = dict(tolerance_multiple=mult, Exp=int(eid), n_matches=int(ok.sum()))
          if ok.sum() >= 5:
              g = lhs[ok]; span = (np.percentile(g, 95, 0) - np.percentile(g, 5, 0)) / (BOUNDS[:, 1] - BOUNDS[:, 0])
              r.update({f'span_{p}': float(s) for p, s in zip(PS, span)})
          nu_rows.append(r)
  NU = pd.DataFrame(nu_rows); NU.to_csv(os.path.join(OUT, 'nonuniqueness.csv'), index=False)
  summ = {str(m): dict(n_targets=int(g[f'span_{PS[0]}'].notna().sum()), of=len(g),
                       median_span={p: float(g[f'span_{p}'].median()) for p in PS}) for m, g in NU.groupby('tolerance_multiple')}
  save('sensitivity.json', dict(sobol=sob, sobol_evaluations=int(len(Xs)), nonuniqueness=summ, lhs_samples=200000))
  log('sensitivity done', json.dumps(summ)[:400])
# random targets vs what the surrogate can reach
if 'reach' in ARGS and not have('reachability.json'):
  from scipy.spatial import cKDTree
  lhs = qmc.scale(qmc.LatinHypercube(d=4, seed=3).random(200000), BOUNDS[:, 0], BOUNDS[:, 1])
  Yl = np.vstack([gp_all.predict(lhs[i:i + 10000]) for i in range(0, len(lhs), 10000)])
  tree = cKDTree(Yl / RES_SD)
  lo, hi = np.array([0.75, 0.52, 0.69]), np.array([1.1, 0.65, 0.85])   # same ranges as the LightTools check targets
  R = lo + (hi - lo) * np.random.default_rng(0).random((20000, 3)); d, _ = tree.query(R / RES_SD)
  dt, _ = tree.query(Fin / RES_SD)
  dmax, _ = tree.query(R / RES_SD, p=np.inf)   # per metric
  save('reachability.json', dict(n_random_targets=20000, ranges=dict(L_ave=[0.75, 1.1], U0=[0.52, 0.65], U_l=[0.69, 0.85]),
       frac_beyond_3sd=float((d > 3).mean()), frac_beyond_2sd=float((d > 2).mean()), frac_beyond_3sd_per_metric=float((dmax > 3).mean()),
       in_range_targets_beyond_3sd=int((dt > 3).sum()), n_in_range_targets=len(Fin),
       rule='euclidean distance in units of GP residual SD to the closest of the GP-predicted metrics of the 200,000 parameter combinations (GP fit on all 496, no simulations). per_metric uses the largest single-metric difference'))
  log('reachability: fraction of random targets beyond 3 residual SD', round(float((d > 3).mean()), 3))
log('ALL DONE')


# designs from inverse training 0 for the 30 LightTools check targets
if 'lt30' in ARGS and not have('lighttools_check_designs.csv'):
  if not have('inverse.json'): sys.exit('run inv first')
  INV_PAT = json.load(open(os.path.join(OUT, 'inverse_patience_selection.json')))['chosen']
  lum = pd.read_excel(os.path.join(HERE, 'historical', 'pred40.xlsx'), sheet_name='Luminance Data')
  T30 = lum[M].values[:30]   # first 30 of the 40
  f_, _ = fit_member(Ftr, Ttr, INV_CFG, INV_SEED0, scaler=MinMaxScaler, drop=INV_DROPOUT, lr_pat=INV_PAT)
  # has to be the same net as training 0
  same = np.abs(np.clip(f_(Fin), BOUNDS[:, 0], BOUNDS[:, 1]) - pd.read_csv(os.path.join(OUT, 'inverse_in_member0.csv'))[[f'pred_{s}' for s in PS]].values).max()
  if same > 1e-4: sys.exit(f'not the same net as training 0 (diff {same})')
  G = np.clip(f_(T30), BOUNDS[:, 0], BOUNDS[:, 1])
  excl = set(jte) | set(top)
  ev = gpr(4).fit(full.drop(index=[i for i in excl if i in full.index])[P].values, full.drop(index=[i for i in excl if i in full.index])[M].values)
  S = ev.predict(G)
  out = pd.DataFrame({'target_no': lum['Exp'].values[:30], **{s: G[:, j] for j, s in enumerate(PS)},
                      **{f'target_{s}': T30[:, j] for j, s in enumerate(MS)}, **{f'surrogate_{s}': S[:, j] for j, s in enumerate(MS)}})
  out.to_csv(os.path.join(OUT, 'lighttools_check_designs.csv'), index=False)
  log('lt30 done, diff to training 0:', same)
