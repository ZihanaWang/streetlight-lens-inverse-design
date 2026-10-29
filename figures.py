# figures 4 to 9. python figures.py [arch|sens|forward|inverse|heldout|all]
# pdf + png go to figures_v2/ (or $FIG_OUT), drawn at print size in 9 pt Liberation Serif like the captions
import sys, json, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch
from matplotlib.transforms import offset_copy
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
from common import load, composite_top, P, M, PS, MS, BOUNDS

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.environ.get('RESULTS_DIR', os.path.join(HERE, 'results_v2'))
OUT = os.environ.get('FIG_OUT', os.path.join(HERE, 'figures_v2'))
J = lambda n: json.load(open(os.path.join(RES, n)))
TW = 5.27   # text width, inches
LBL_OOD = 'OOD designs (30)'
N_REST = len(load(2)) - 30
LBL_REST = f'Training and in-range test designs ({N_REST})'

INK, SLATE, MUTED = '#222222', '#2e3a4b', '#6b7280'
TEAL, TEAL_L, TEAL_D = '#3c7f8c', '#d7e8eb', '#24525b'
AMBER, AMBER_L, AMBER_D = '#d9962b', '#f5e1b8', '#8a5a12'
FS = 9
FS_BOX = 8.5   # inside boxes
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Liberation Serif', 'Times New Roman'],
                     'mathtext.fontset': 'custom', 'mathtext.rm': 'Liberation Serif', 'mathtext.it': 'Liberation Serif:italic',
                     'mathtext.bf': 'Liberation Serif:bold', 'font.size': FS, 'axes.labelsize': FS, 'axes.labelpad': 2.5,
                     'xtick.labelsize': FS, 'ytick.labelsize': FS, 'axes.edgecolor': SLATE, 'axes.linewidth': 0.5,
                     'axes.labelcolor': INK, 'xtick.color': SLATE, 'ytick.color': SLATE, 'xtick.labelcolor': INK, 'ytick.labelcolor': INK,
                     'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True,
                     'xtick.minor.visible': True, 'ytick.minor.visible': True, 'xtick.major.size': 3, 'ytick.major.size': 3,
                     'xtick.minor.size': 1.2, 'ytick.minor.size': 1.2, 'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
                     'xtick.minor.width': 0.35, 'ytick.minor.width': 0.35, 'xtick.major.pad': 2.5, 'ytick.major.pad': 2.5,
                     'legend.frameon': False, 'legend.fontsize': FS, 'lines.linewidth': 0.8, 'axes.unicode_minus': True,
                     'pdf.fonttype': 42, 'savefig.dpi': 600, 'savefig.pad_inches': 0.02})
UNIT = {'L_ave': 'cd/m$^2$', 'U0': '', 'U_l': '', 'R1': 'mm', 'R2': 'mm', 'H': 'mm', 'theta': 'deg'}
SYM = {'L_ave': r'$L_{ave}$', 'U0': r'$U_0$', 'U_l': r'$U_l$', 'R1': r'$R_1$', 'R2': r'$R_2$', 'H': r'$H$', 'theta': r'$\theta$'}
LAB = {k: SYM[k] + (f' ({u})' if u else '') for k, u in UNIT.items()}
MS_IN = dict(s=2.6, facecolor=TEAL, edgecolor=TEAL_D, linewidth=0.15, alpha=0.85)
MS_HO = dict(s=4.2, marker='^', facecolor=AMBER, edgecolor=AMBER_D, linewidth=0.2)   # OOD designs


def save(fig, name):
    fig.savefig(f'{OUT}/{name}.pdf', bbox_inches='tight'); fig.savefig(f'{OUT}/{name}.png', bbox_inches='tight'); plt.close(fig)


def panel(ax, letter, title=None, dx=-22):
    # (a), (b), ... left of the axes, title next to it
    tr = offset_copy(ax.transAxes, fig=ax.figure, x=dx, y=3, units='points')
    ax.text(0, 1, f'({letter})', transform=tr, fontsize=FS, color=INK, va='bottom', ha='left')
    if title:
        tr2 = offset_copy(ax.transAxes, fig=ax.figure, x=dx + 15, y=3.4, units='points')
        ax.text(0, 1, title, transform=tr2, fontsize=FS, color=INK, va='bottom', ha='left')


def corner(ax, text, where='auto', avoid=()):
    # where='auto' picks the corner with the fewest points
    if where == 'auto':
        pts = np.vstack([ax.transAxes.inverted().transform(ax.transData.transform(np.column_stack(xy))) for xy in ax._pts]) if getattr(ax, '_pts', None) else np.empty((0, 2))
        nl = text.count('\n') + 1; w, h = 0.5, 0.06 + 0.1 * nl
        boxes = {'tl': (0, 1 - h, w, 1), 'br': (1 - w, 0, 1, h), 'tr': (1 - w, 1 - h, 1, 1), 'bl': (0, 0, w, h)}
        cnt = {c: int(np.sum((pts[:, 0] >= b[0]) & (pts[:, 0] <= b[2]) & (pts[:, 1] >= b[1]) & (pts[:, 1] <= b[3]))) for c, b in boxes.items() if c not in avoid}
        where = min(cnt, key=lambda c: (cnt[c], list(boxes).index(c)))
    x, ha = (0.05, 'left') if where[1] == 'l' else (0.95, 'right')
    y, va = (0.95, 'top') if where[0] == 't' else (0.05, 'bottom')
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=FS, color=INK, linespacing=1.25, zorder=6)
    return where


def parity(ax, y, p, lims=None, nticks=4, **kw):
    lo, hi = lims if lims else (min(np.min(y), np.min(p)), max(np.max(y), np.max(p)))
    pad = 0.05 * (hi - lo); lo, hi = lo - pad, hi + pad
    xx = np.array([lo, hi])
    ax.plot(xx, xx, color=SLATE, lw=0.6, ls=(0, (4, 2.5)), zorder=1)
    ax.scatter(y, p, zorder=3, **{**MS_IN, **kw}); ax._pts = [(np.asarray(y), np.asarray(p))]
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect('equal', adjustable='box')
    ax.xaxis.set_major_locator(MaxNLocator(nticks)); ax.yaxis.set_major_locator(MaxNLocator(nticks))


def fmt_r2(x): return f'{x:.2f}'.replace('-', '−')


# fig 4
def arch():
    ch = J('forward_test.json')['chosen']; ens = ch.startswith('C')
    fig, axes = plt.subplots(2, 1, figsize=(TW, 3.25), gridspec_kw=dict(hspace=0.06, left=0.0, right=1.0, top=1.0, bottom=0.0))
    rows = [dict(ax=axes[0], tag='a', title='Forward model', ins=['$R_1$', '$R_2$', '$H$', r'$\theta$'], outs=[r'$L_{ave}$', '$U_0$', '$U_l$'],
                 scale='Input\nscaling', last=('Mean of 5' if ens else None), nout=3, drop=False, frame=ens),
            dict(ax=axes[1], tag='b', title='Inverse model', ins=[r'$L_{ave}$', '$U_0$', '$U_l$'], outs=['$R_1$', '$R_2$', '$H$', r'$\theta$'],
                 scale='Input\nscaling', last='Clip to\nbounds', nout=4, drop=True, frame=False)]
    R = 0.17; yc = 2.4
    for r in rows:
        ax = r['ax']; ax.set_xlim(-0.3, 18.9); ax.set_ylim(-0.1, 5.75); ax.set_aspect('equal'); ax.axis('off')
        ax.text(-0.3, 5.7, f"({r['tag']})", fontsize=FS, va='top'); ax.text(0.55, 5.68, r['title'], fontsize=FS, va='top')
        node = lambda x, y, fc, ec, ls='-', z=3: ax.add_patch(Circle((x, y), R, fc=fc, ec=ec, lw=0.6, ls=ls, zorder=z))
        box = lambda x0, w, txt, fc, ec: (ax.add_patch(FancyBboxPatch((x0, yc - 0.55), w, 1.1, boxstyle='round,pad=0.02,rounding_size=0.06',
                                                                      fc=fc, ec=ec, lw=0.6, zorder=2)),
                                          ax.text(x0 + w / 2, yc, txt, ha='center', va='center', fontsize=FS_BOX, linespacing=1.1, zorder=3))
        arrow = lambda x0, x1: ax.annotate('', xy=(x1, yc), xytext=(x0, yc), arrowprops=dict(arrowstyle='-|>', color=SLATE, lw=0.55, mutation_scale=7))
        # inputs
        inp = []
        for y, t in zip(np.linspace(3.6, 1.2, len(r['ins'])), r['ins']):
            node(1.15, y, 'white', SLATE); inp.append((1.15, y)); ax.text(0.85, y, t, ha='right', va='center', fontsize=FS)
        box(1.85, 2.0, r['scale'], AMBER_L, AMBER_D)
        for p in inp: ax.plot([p[0] + R, 1.85], [p[1], yc], color=SLATE, lw=0.4, zorder=1)
        # dense layers
        def col(x, n=None):
            pts = []
            ys = [4.2, 3.6, 3.0, None, 1.2, 0.6] if n is None else list(np.linspace(3.6, 1.2, n))
            for y in ys:
                if y is None:
                    for dy in (-0.22, 0, 0.22): ax.add_patch(Circle((x, 2.1 + dy), 0.035, fc=MUTED, ec='none'))
                    continue
                node(x, y, TEAL_L, TEAL_D); pts.append((x, y))
            return pts
        xs = (5.15, 6.85, 8.55, 10.25)
        L = [col(x) for x in xs[:3]] + [col(xs[3], r['nout'])]
        if r['drop']:
            for x in xs[:2]: ax.add_patch(Circle((x, 3.0), R, fc='white', ec=AMBER_D, lw=0.9, ls=(0, (2.6, 2.0)), zorder=4))
        for x, n in zip(xs, (64, 32, 16, r['nout'])): ax.text(x, 4.6, f'{n}', ha='center', fontsize=FS)
        for q in L[0]: ax.plot([3.85, q[0] - R], [yc, q[1]], color=TEAL, lw=0.25, alpha=0.5, zorder=1)
        for A, B in zip(L[:-1], L[1:]):
            for p in A:
                for q in B: ax.plot([p[0] + R, q[0] - R], [p[1], q[1]], color=TEAL, lw=0.25, alpha=0.35, zorder=1)
        box(11.0, 2.25, 'Output\nrescaling', AMBER_L, AMBER_D)
        for p in L[-1]: ax.plot([p[0] + R, 11.0], [p[1], yc], color=SLATE, lw=0.4, zorder=1)
        x_out = 13.25
        if r['frame']:
            ax.add_patch(FancyBboxPatch((1.6, 0.25), 11.9, 4.85, boxstyle='round,pad=0,rounding_size=0.08', fc='none', ec=TEAL, lw=0.55, ls=(0, (3, 2)), zorder=0))
            ax.text(13.4, 5.0, '×5', ha='right', va='top', fontsize=FS, color=TEAL_D)
        if r['last']:
            x0 = 14.15; arrow(x_out + (0.25 if r['frame'] else 0.0), x0 - 0.03)
            if r['tag'] == 'a': box(x0, 2.35, r['last'], TEAL_L, TEAL_D)
            else: box(x0, 2.35, r['last'], AMBER_L, AMBER_D)
            x_out = x0 + 2.35
        arrow(x_out + 0.03, 17.15)
        ax.text(17.75, yc, '\n'.join(r['outs']), ha='center', va='center', fontsize=FS, linespacing=1.45)
    save(fig, 'architecture')



# fig 6
def forward():
    d = pd.read_csv(os.path.join(RES, 'forward_test_predictions.csv'), index_col=0); r = J('forward_test.json')['dnn']
    fig, axes = plt.subplots(1, 3, figsize=(TW, 2.2), gridspec_kw=dict(wspace=0.32, left=0.1, right=0.98, bottom=0.2, top=0.86))
    for k, (ax, s, t) in enumerate(zip(axes, MS, 'abc')):
        y, p = d[f'true_{s}'].values, d[f'pred_{s}'].values
        parity(ax, y, p)
        txt = f"$R^2$ = {fmt_r2(r[s]['R2'])}\nMAPE = {r[s]['MAPE']:.1f}%"
        corner(ax, txt, 'tl')
        u = f' ({UNIT[s]})' if UNIT[s] else ''
        ax.set_xlabel('Simulated' + u)
        if k == 0: ax.set_ylabel('Predicted' + u)
        panel(ax, t, SYM[s], dx=-30 if k == 0 else -16)
    save(fig, 'forward_parity')


# figs 8 and 9
def inverse():
    a = pd.read_csv(os.path.join(RES, 'inverse_in_member0.csv'), index_col=0); h = pd.read_csv(os.path.join(RES, 'inverse_ho_member0.csv'), index_col=0)
    inv = J('inverse.json')
    fig, axs = plt.subplots(2, 2, figsize=(4.1, 4.0), gridspec_kw=dict(wspace=0.42, hspace=0.5, left=0.15, right=0.97, bottom=0.1, top=0.86))
    for k, (ax, s, bnd) in enumerate(zip(axs.flat, PS, BOUNDS)):
        parity(ax, a[f'true_{s}'].values, a[f'pred_{s}'].values, lims=tuple(bnd), label='In-range test designs (68)')
        ax.scatter(h[f'true_{s}'], h[f'pred_{s}'], zorder=4, label=LBL_OOD, **MS_HO)
        corner(ax, f"$R^2$ = {fmt_r2(inv['param_in'][s]['R2'])}")
        ax.set_xlabel(f'True ({UNIT[s]})'); ax.set_ylabel(f'Predicted ({UNIT[s]})')
        panel(ax, 'abcd'[k], SYM[s], dx=-32)
    hd, lb = axs[0, 0].get_legend_handles_labels()
    fig.canvas.draw(); ytop = axs[0, 0].get_position().y1
    fig.legend(hd, lb, loc='lower center', ncol=2, bbox_to_anchor=(0.56, ytop + 0.05), handletextpad=0.3, columnspacing=1.6, markerscale=2.2)
    save(fig, 'inverse_parity')
    sp = pd.read_csv(os.path.join(RES, 'inverse_surrogate_predictions.csv')); m0 = sp[(sp.source == 'inverse training 0') & (sp.set == 'in')]
    fig, axes = plt.subplots(1, 3, figsize=(TW, 2.2), gridspec_kw=dict(wspace=0.32, left=0.1, right=0.98, bottom=0.2, top=0.86))
    for k, (ax, s) in enumerate(zip(axes, MS)):
        parity(ax, m0[f'target_{s}'].values, m0[f'surrogate_{s}'].values)
        corner(ax, f"MAPE = {inv['surrogate_perf_in'][s]['MAPE']:.1f}%")
        u = f' ({UNIT[s]})' if UNIT[s] else ''
        ax.set_xlabel(f'Target{u}')
        if k == 0: ax.set_ylabel(f'Reached, surrogate\nestimate{u}')
        panel(ax, 'abc'[k], SYM[s], dx=-30 if k == 0 else -16)
    save(fig, 'inverse_reached')


# fig 7
def sens():
    lamp = LinearSegmentedColormap.from_list('lamp', ['#ffffff', AMBER_L, AMBER, AMBER_D])
    sj = J('sensitivity.json')['sobol']
    ST = np.array([[sj[m][p]['ST'] for p in PS] for m in MS]); STc = np.array([[sj[m][p]['ST_conf'] for p in PS] for m in MS])
    nu = pd.read_csv(os.path.join(RES, 'nonuniqueness.csv')); nu = nu[nu.tolerance_multiple == 1.0].dropna()
    fig = plt.figure(figsize=(TW, 2.45)); gs = fig.add_gridspec(1, 3, width_ratios=[1, 0.05, 1.15], wspace=0.12)
    a = fig.add_subplot(gs[0, 0]); cax = fig.add_subplot(gs[0, 1]); b = fig.add_subplot(gs[0, 2])
    im = a.imshow(ST, cmap=lamp, vmin=0, vmax=1, aspect='auto')
    for k in range(1, 4): a.axvline(k - 0.5, color='white', lw=1.2)
    for k in range(1, 3): a.axhline(k - 0.5, color='white', lw=1.2)
    for sp_ in a.spines.values(): sp_.set_visible(False)
    a.set_xticks(range(4)); a.set_xticklabels([SYM[p] for p in PS], fontsize=FS)
    a.set_yticks(range(3)); a.set_yticklabels([SYM[m] for m in MS], fontsize=FS)
    for i in range(3):
        for j in range(4):
            dark = ST[i, j] > 0.6
            a.text(j, i, f'{ST[i, j]:.2f}', ha='center', va='center', fontsize=FS, color='white' if dark else INK)
    a.tick_params(length=0, which='both', top=False, right=False, bottom=False, left=False, pad=3); a.minorticks_off()
    panel(a, 'a', 'Total-order Sobol index', dx=-26)
    cb = fig.colorbar(im, cax=cax, ticks=[0, 0.5, 1.0]); cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=FS, direction='in', length=1.5, right=True); cb.ax.minorticks_off()
    pos = cax.get_position(); cax.set_position([pos.x0 - 0.035, pos.y0, pos.width * 0.6, pos.height])
    data = [nu[f'span_{p}'].values * 100 for p in PS]
    bp = b.boxplot(data, widths=0.55, patch_artist=True, showfliers=False,
                   medianprops=dict(color=INK, lw=0.9), whiskerprops=dict(color=SLATE, lw=0.5), capprops=dict(color=SLATE, lw=0.5))
    rng = np.random.default_rng(0)
    for k, box in enumerate(bp['boxes']):
        hi = PS[k] == 'R2'
        box.set(facecolor=AMBER_L if hi else TEAL_L, edgecolor=AMBER_D if hi else TEAL_D, linewidth=0.55)
        b.scatter(k + 1 + rng.uniform(-0.14, 0.14, len(data[k])), data[k], s=1.6, color=AMBER_D if hi else TEAL_D, alpha=0.55, linewidth=0, zorder=3)
    b.tick_params(axis='x', which='minor', bottom=False, top=False)
    b.set_xticks(range(1, 5)); b.set_xticklabels([SYM[p] for p in PS], fontsize=FS)
    b.set_ylabel('Span among matches\n(5th to 95th percentile, % of range)'); b.set_ylim(0, 100)
    pb = b.get_position(); b.set_position([pb.x0 + 0.115, pb.y0, pb.width - 0.115, pb.height])   # room for the colorbar
    panel(b, 'b', f'Surrogate matches per target ({len(nu)} targets)', dx=-36)
    save(fig, 'sensitivity')


# fig 5
def heldout():
    ef = load(2); top, sc = composite_top(ef, 30); rest = ef.drop(top); best = list(top[:2])
    ul_max = rest[M[2]].max()
    fig = plt.figure(figsize=(TW, 3.75)); gs = fig.add_gridspec(2, 2, wspace=0.36, hspace=0.52)
    ring = dict(s=38, facecolor='none', edgecolor=INK, linewidth=0.6, zorder=5)
    line = dict(color=SLATE, lw=0.55, ls=(0, (4, 2.2)), zorder=1)
    for k, (i, j, t) in enumerate([(0, 1, 'a'), (0, 2, 'b'), (1, 2, 'c')]):
        ax = fig.add_subplot(gs[k // 2, k % 2])
        ax.scatter(rest[M[i]], rest[M[j]], s=2.6, color=TEAL, alpha=0.65, linewidth=0, label=LBL_REST, zorder=2)
        ax.scatter(ef.loc[top, M[i]], ef.loc[top, M[j]], label=LBL_OOD, zorder=3, **MS_HO)
        ax.scatter(ef.loc[best, M[i]], ef.loc[best, M[j]], **ring)
        if j == 2: ax.axhline(ul_max, **line)   # highest U_l among the rest
        ax.set_xlabel(LAB[MS[i]]); ax.set_ylabel(LAB[MS[j]]); panel(ax, t, dx=-32)
        ax.xaxis.set_major_locator(MaxNLocator(5)); ax.yaxis.set_major_locator(MaxNLocator(5))
    # (d) score vs rank, log x so the top 30 aren't squeezed together
    ax = fig.add_subplot(gs[1, 1]); ss = sc.sort_values(ascending=False); r = np.arange(1, len(ss) + 1); ho = ss.index.isin(top)
    ax.scatter(r[~ho], ss.values[~ho], s=2.6, color=TEAL, alpha=0.65, linewidth=0, zorder=2)
    ax.scatter(r[ho], ss.values[ho], zorder=3, **MS_HO)
    ax.scatter(r[:2], ss.values[:2], **ring)
    ax.axvline(30.5, ymin=0.5, **line)
    ax.set_xscale('log'); ax.set_xlim(0.75, 520); ax.set_ylim(0.0, 0.75)
    ax.set_xticks([1, 10, 100]); ax.set_xticklabels(['1', '10', '100'])
    ax.set_xlabel('Rank by composite score'); ax.set_ylabel('Composite score'); panel(ax, 'd', dx=-32)
    ax.yaxis.set_major_locator(MaxNLocator(4))
    for k, f in enumerate(('render_a.png', 'render_b.png')):   # renderings in the empty lower left corner
        ins = ax.inset_axes([0.03 + k * 0.355, 0.05, 0.34, 0.42]); ins.imshow(plt.imread(os.path.join(HERE, 'assets', f)))
        ins.set_xticks([]); ins.set_yticks([]); ins.minorticks_off()
        for sp in ins.spines.values(): sp.set_linewidth(0.4); sp.set_edgecolor(MUTED)
    for k, (x_from, rank) in enumerate([(0.12, 1), (0.46, 2)]):   # left = rank 1, right = rank 2
        ax.annotate('', xy=(rank, ss.values[rank - 1]), xytext=(x_from, 0.47), textcoords='axes fraction',
                    arrowprops=dict(arrowstyle='-|>', color=INK, lw=0.6, mutation_scale=7, shrinkA=0, shrinkB=6), zorder=6)
    hd, lb = fig.axes[0].get_legend_handles_labels()
    fig.canvas.draw(); ytop = fig.axes[0].get_position().y1   # legend above a and b
    fig.legend(hd, lb, loc='lower center', ncol=2, bbox_to_anchor=(0.53, ytop + 0.03), handletextpad=0.3, columnspacing=2.0, markerscale=2.2)
    save(fig, 'heldout_distribution')


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    w = sys.argv[1] if len(sys.argv) > 1 else 'all'
    for name, f in [('arch', arch), ('sens', sens), ('forward', forward), ('inverse', inverse), ('heldout', heldout)]:
        if w in (name, 'all'): f(); print('made', name)
