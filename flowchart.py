# fig 3 (workflow). python flowchart.py -> figures_v2/flowchart.svg and .png
# needs playwright + chromium for the png
import os
from playwright.sync_api import sync_playwright

OUT = os.environ.get('FIG_OUT', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures_v2'))
W, H = 880, 664
FONT = "Helvetica, Arial, 'Liberation Sans', sans-serif"
BLUE = '#0000ff'
S = []


def text(cx, cy, lines, color='#000', size=11, italic=False, anchor='middle', weight='normal'):
    lh = size * 1.22
    y0 = cy - (len(lines) - 1) * lh / 2 + size * 0.36
    st = 'font-style:italic;' if italic else ''
    for i, ln in enumerate(lines):
        S.append(f'<text x="{cx}" y="{y0 + i * lh:.1f}" text-anchor="{anchor}" font-family="{FONT}" font-size="{size}" '
                 f'font-weight="{weight}" fill="{color}" style="{st}">{ln}</text>')


def box(x, y, w, h, lines, kind, color='#000'):
    # kind = teal, yellow, solid (SolidWorks), light (LightTools), process, ml or eval
    st = dict(teal=('url(#gTeal)', '#4d4d4d', 1.2), yellow=('url(#gYellow)', '#666666', 1.2),
              solid=('#fff2cc', '#d6b656', 2), light=('#d5e8d4', '#82b366', 2.2), process=('url(#gGray)', '#4d4d4d', 1.2),
              ml=('#f8cecc', '#b85450', 2.2), eval=('#ffffff', '#666666', 2))[kind]
    S.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="7" fill="{st[0]}" stroke="{st[1]}" stroke-width="{st[2]}"/>')
    if kind == 'process':
        for dx in (7, w - 7):
            S.append(f'<line x1="{x + dx}" y1="{y}" x2="{x + dx}" y2="{y + h}" stroke="#4d4d4d" stroke-width="1.2"/>')
    text(x + w / 2, y + h / 2, lines, color)
    return dict(l=(x, y + h / 2), r=(x + w, y + h / 2), t=(x + w / 2, y), b=(x + w / 2, y + h), x=x, y=y, w=w, h=h)


def block_arrow(x0, x1, cy, h=22):
    sh = h * 0.42; hw = min(14, (x1 - x0) * 0.5)
    pts = [(x0, cy - sh / 2), (x1 - hw, cy - sh / 2), (x1 - hw, cy - h / 2), (x1, cy), (x1 - hw, cy + h / 2), (x1 - hw, cy + sh / 2), (x0, cy + sh / 2)]
    S.append('<polygon points="' + ' '.join(f'{a:.1f},{b:.1f}' for a, b in pts) + '" fill="#fff" stroke="#000" stroke-width="2"/>')


def between(a, b, pad=6):
    block_arrow(a['r'][0] + pad, b['l'][0] - pad, a['r'][1])


def line(pts, arrow=True):
    d = 'M ' + ' L '.join(f'{x:.1f} {y:.1f}' for x, y in pts)
    S.append(f'<path d="{d}" fill="none" stroke="#000" stroke-width="2.6" stroke-linejoin="miter"' + (' marker-end="url(#ah)"' if arrow else '') + '/>')


def panel(x, y, w, h, fill, stroke):
    S.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="22" fill="{fill}" stroke="{stroke}" stroke-width="1.3"/>')


def vs(x, y):
    text(x, y, ['vs.'], size=12, weight='bold')


# background panels
panel(4, 3, 680, 230, '#f8d3d0', '#b85450')
panel(52, 245, 824, 172, '#b1ddf0', '#10739e')
panel(4, 430, 690, 230, '#eef7dc', '#333333')

# simulation
y1, h1 = 20, 54
a = box(20, y1, 112, h1, ['Structural', 'Parameters', '(R₁, R₂, H, θ)'], 'teal')
b = box(176, y1, 96, h1, ['SolidWorks'], 'solid')
c = box(312, y1, 66, h1, ['Lens', 'Model', '(IGES)'], 'process')
d = box(418, y1, 96, h1, ['LightTools'], 'light', BLUE)
e = box(556, y1, 114, h1, ['LightTools', 'Simulated', 'Luminance', 'Metrics'], 'yellow', BLUE)
for p, q in [(a, b), (b, c), (c, d), (d, e)]:
    between(p, q)
db_x, db_y, db_w, db_h = 252, 118, 136, 96
S.append(f'<path d="M {db_x} {db_y + 10} L {db_x} {db_y + db_h - 10} A {db_w / 2} 10 0 0 0 {db_x + db_w} {db_y + db_h - 10} '
         f'L {db_x + db_w} {db_y + 10} A {db_w / 2} 10 0 0 0 {db_x} {db_y + 10} Z" fill="#dae8fc" stroke="#6c8ebf" stroke-width="1.4"/>')
S.append(f'<ellipse cx="{db_x + db_w / 2}" cy="{db_y + 10}" rx="{db_w / 2}" ry="10" fill="#dae8fc" stroke="#6c8ebf" stroke-width="1.4"/>')
text(db_x + db_w / 2, db_y + 56, ['Simulated Dataset', '(496 designs;', '366 in inverse-', 'model dataset)'])
line([a['b'], (a['b'][0], 98), (e['b'][0], 98), e['b']], arrow=False)
line([(db_x + db_w / 2, 98), (db_x + db_w / 2, db_y - 1)])
S.append(f'<rect x="402" y="102" width="176" height="44" fill="#ffffff" opacity="0.85"/>')
text(406, 124, ['Store every simulated design.', 'Flag designs that pass the', 'SLU Tool merit function.'], size=10, italic=True, anchor='start')

# forward model test, inverse model
y2, y3, h2 = 262, 340, 58
f = box(72, y2, 116, h2, ['Structural', 'Parameters', '(set in SolidWorks)'], 'teal', BLUE)
g = box(234, y2, 106, h2, ['ML', 'Forward Model'], 'ml')
hh = box(386, y2, 108, h2, ['ML Predicted', 'Luminance', 'Metrics'], 'yellow')
vs(522, y2 + h2 / 2)
i = box(550, y2, 112, h2, ['LightTools', 'Simulated', 'Luminance', 'Metrics'], 'yellow', BLUE)
j = box(708, y2, 152, h2, ['Evaluate Forward', 'Model Performance'], 'eval')
for p, q in [(f, g), (g, hh), (i, j)]:
    between(p, q)
k = box(72, y3, 116, h2, ['Target', 'Luminance', 'Metrics'], 'yellow')
l_ = box(234, y3, 106, h2, ['ML', 'Inverse Model'], 'ml')
m = box(386, y3, 108, h2, ['Proposed', 'Structural', 'Parameters'], 'teal')
for p, q in [(k, l_), (l_, m)]:
    between(p, q)
# dataset feeds both models
line([(db_x, db_y + 48), (40, db_y + 48), (40, f['l'][1]), (f['l'][0] - 1, f['l'][1])])
line([(40, f['l'][1]), (40, k['l'][1]), (k['l'][0] - 1, k['l'][1])])

# inverse model test
y4, y5, y6, h3 = 446, 522, 598, 50
n = box(24, y4, 116, h3, ['Proposed', 'Structural', 'Parameters'], 'teal')
o = box(176, y4, 96, h3, ['SolidWorks'], 'solid')
p_ = box(312, y4, 66, h3, ['Lens', 'Model', '(IGES)'], 'process')
q = box(418, y4, 96, h3, ['LightTools'], 'light', BLUE)
r = box(556, y4, 124, h3, ['LightTools Simulated', 'Luminance Metrics', '(30 random targets)'], 'yellow', BLUE)
for u, v in [(n, o), (o, p_), (p_, q), (q, r)]:
    between(u, v)
sg = box(176, y5, 202, h3, ['Independent Gaussian', 'Process Surrogate'], 'ml')
se = box(418, y5, 160, h3, ['Surrogate-Estimated', 'Luminance Metrics', '(68 in-range targets)'], 'yellow')
between(sg, se)
line([(n['b'][0], n['b'][1]), (n['b'][0], sg['l'][1]), (sg['l'][0] - 1, sg['l'][1])])
tg = box(24, y6, 116, h3, ['Target', 'Luminance', 'Metrics'], 'yellow')
vs(160, y6 + h3 / 2)
rc = box(184, y6, 140, h3, ['Reached Luminance', 'Metrics'], 'yellow')
ev = box(372, y6, 152, h3, ['Evaluate Inverse', 'Model Performance'], 'eval')
between(rc, ev)
# connectors between the stages
line([m['b'], (m['b'][0], 424), (16, 424), (16, n['l'][1]), (n['l'][0] - 1, n['l'][1])])
line([(k['l'][0], k['l'][1] + 16), (8, k['l'][1] + 16), (8, tg['l'][1]), (tg['l'][0] - 1, tg['l'][1])], arrow=True)
line([r['b'], (r['b'][0], 582), (rc['t'][0] + 30, 582), (rc['t'][0] + 30, rc['t'][1] - 1)])
line([se['b'], (se['b'][0], 576), (rc['t'][0] - 10, 576), (rc['t'][0] - 10, rc['t'][1] - 1)])

svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">
<defs>
<linearGradient id="gTeal" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#ffffff"/><stop offset="1" stop-color="#67ab9f"/></linearGradient>
<linearGradient id="gYellow" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#ffff88"/><stop offset="1" stop-color="#ffffff"/></linearGradient>
<linearGradient id="gGray" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#f5f5f5"/><stop offset="1" stop-color="#9e9e9e"/></linearGradient>
<marker id="ah" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="4.2" markerHeight="4.2" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#000"/></marker>
</defs>
<rect width="{W}" height="{H}" fill="#ffffff"/>
{chr(10).join(S)}
</svg>'''
os.makedirs(OUT, exist_ok=True)
open(os.path.join(OUT, 'flowchart.svg'), 'w').write(svg)
with sync_playwright() as pw:
    br = pw.chromium.launch(); pg = br.new_page(viewport=dict(width=W, height=H), device_scale_factor=4)
    pg.set_content(f'<html><body style="margin:0">{svg}</body></html>'); pg.screenshot(path=os.path.join(OUT, 'flowchart.png'), full_page=False)
    br.close()
print('saved', OUT)
