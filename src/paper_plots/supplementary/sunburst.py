# sunburst.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties
from matplotlib.colors import to_rgb
from functools import lru_cache

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11

# =============================================================================
# CONFIGURATION
# =============================================================================
RING_WIDTH       = 0.55
INNER_RADIUS     = 0.90
NUM_LAYERS       = 4
LAYER_FONT       = {1: 22, 2: 17, 3: 14, 4: 14}   # layer‑4 bumped (radial ⇒ more room)
MIN_FONT         = 4.0
MIN_METHOD_FRAC  = 0.12

# ── replace the old threshold block ──────────────────────────────
NOMINAL = 0.95

# Undercoverage thresholds (below nominal)
THRESH_UNDER_GREEN = 0.92     # 0.92–0.95 : green gradient
THRESH_UNDER_ORANGE = 0.90    # 0.90–0.92 : orange gradient
THRESH_UNDER_RED = 0.85       # 0.85–0.90 : red gradient  / <0.85 solid red

# Overcoverage thresholds (above nominal – more lenient, less room)
THRESH_OVER_GREEN = 0.98      # 0.95–0.98 : green gradient
THRESH_OVER_ORANGE = 0.99     # 0.98–0.99 : orange gradient
THRESH_OVER_RED = 1.00        # 0.99–1.00 : red gradient  / >1.00 solid red

_GS = np.array(to_rgb("#2e7d32"))
_GL = np.array(to_rgb("#a5d6a7"))
_OR = np.array(to_rgb("#e65100"))
_RD = np.array(to_rgb("#c62828"))

DISPLAY = {
    'basic': 'Basic', 'bca': 'BCa', 'percentile': 'Percentile',
    'param_t': 'Param t', 'wilson': 'Wilson',
    'classification': 'Classif.', 'segmentation': 'Segm.',
    'micro': 'Micro', 'macro': 'Macro',
    'mean': 'Mean', 'median': 'Median',
    'accuracy': 'Acc.', 'balanced_accuracy': 'BA',
    'auc': 'AUC', 'ap': 'AP',
    'f1': 'F1', 'f1_score': 'F1', 'mcc': 'MCC',
    'dsc': 'DSC', 'iou': 'IoU', 'cldice': 'clDice',
    'boundary_iou': 'Bound. IoU',
    'hd': 'HD', 'hd_perc': 'HD95', 'hd95': 'HD95',
    'assd': 'ASSD', 'masd': 'MASD', 'nsd': 'NSD',
}


def _pretty(name):
    s = str(name)
    return DISPLAY.get(s, s.replace('_', ' ').title())


# =============================================================================
# COLOUR HELPERS
# =============================================================================
def coverage_to_color(v):
    """Map a coverage value to an RGB tuple.

    The palette is centred on NOMINAL = 0.95 (dark green).
    Undercoverage (v < 0.95) is penalised more aggressively than
    overcoverage (v > 0.95) because the reference is close to 1 and
    conservative intervals are less harmful than anti‑conservative ones.
    """
    if v >= NOMINAL:
        # ── overcoverage side ─────────────────────────────────────
        if v <= THRESH_OVER_GREEN:                        # 0.95 – 0.98
            t = (v - NOMINAL) / (THRESH_OVER_GREEN - NOMINAL)
            return tuple(_GS + t * (_GL - _GS))           # dark→light green
        if v <= THRESH_OVER_ORANGE:                       # 0.98 – 0.99
            t = (v - THRESH_OVER_GREEN) / (THRESH_OVER_ORANGE - THRESH_OVER_GREEN)
            return tuple(_GL + t * (_OR - _GL))            # light green→orange
        if v <= THRESH_OVER_RED:                          # 0.99 – 1.00
            t = (v - THRESH_OVER_ORANGE) / (THRESH_OVER_RED - THRESH_OVER_ORANGE)
            return tuple(_OR + t * (_RD - _OR))            # orange→red
        return tuple(_RD)                                  # > 1.00 (safety)
    else:
        # ── undercoverage side ────────────────────────────────────
        if v >= THRESH_UNDER_GREEN:                       # 0.92 – 0.95
            t = (v - THRESH_UNDER_GREEN) / (NOMINAL - THRESH_UNDER_GREEN)
            return tuple(_GL + t * (_GS - _GL))            # light green→dark green
        if v >= THRESH_UNDER_ORANGE:                      # 0.90 – 0.92
            t = (v - THRESH_UNDER_ORANGE) / (THRESH_UNDER_GREEN - THRESH_UNDER_ORANGE)
            return tuple(_OR + t * (_GL - _OR))            # orange→light green
        if v >= THRESH_UNDER_RED:                         # 0.85 – 0.90
            t = (v - THRESH_UNDER_RED) / (THRESH_UNDER_ORANGE - THRESH_UNDER_RED)
            return tuple(_RD + t * (_OR - _RD))            # red→orange
        return tuple(_RD)                                  # < 0.85


def _text_color(rgb):
    lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    return 'white' if lum < 0.5 else 'black'


def _build_coverage_cmap(n=256):
    lo = THRESH_UNDER_RED - 0.02          # 0.83
    hi = min(THRESH_OVER_RED + 0.005, 1.0)  # 1.00
    vals = np.linspace(lo, hi, n)
    colors = [coverage_to_color(v) for v in vals]
    return mcolors.ListedColormap(colors), lo, hi


# =============================================================================
# TREE NODE
# =============================================================================
class Node:
    def __init__(self, name, layer):
        self.name     = name
        self.layer    = layer
        self.children = []
        self.coverage = None
        self.avg_cov  = None
        self.a0 = self.a1 = 0.0
        self.merged   = False

    def is_leaf(self):
        return not self.children

    def leaf_count(self):
        return 1 if self.is_leaf() else sum(c.leaf_count() for c in self.children)

    def compute_avg(self):
        if self.is_leaf():
            self.avg_cov = self.coverage
            return self.avg_cov
        vals = [c.compute_avg() for c in self.children]
        vals = [v for v in vals if v is not None]
        self.avg_cov = float(np.mean(vals)) if vals else None
        return self.avg_cov

    def sort_recursive(self):
        for c in self.children:
            c.sort_recursive()
        self.children.sort(key=lambda n: n.avg_cov if n.avg_cov is not None else -1)

    def assign_angles(self, start, end):
        self.a0, self.a1 = start, end
        if self.is_leaf():
            return
        total = self.leaf_count()
        span  = (end - start) / total
        cur   = start
        for c in self.children:
            nl = c.leaf_count()
            c.assign_angles(cur, cur + span * nl)
            cur += span * nl


def _assign_root_angles(root, start=0, end=360):
    root.a0, root.a1 = start, end
    children = root.children
    if not children:
        return
    total_angle  = end - start
    leaf_counts  = [c.leaf_count() for c in children]
    total_leaves = sum(leaf_counts)
    fracs = [max(lc / total_leaves, MIN_METHOD_FRAC) for lc in leaf_counts]
    s     = sum(fracs)
    fracs = [f / s for f in fracs]
    cur = start
    for child, frac in zip(children, fracs):
        span = total_angle * frac
        child.assign_angles(cur, cur + span)
        cur += span


# =============================================================================
# BUILD TREE FROM DATAFRAME
# =============================================================================
def build_tree(df):
    root = Node("", 0)
    for method in df['method'].unique():
        m_node = Node(method, 1)
        root.children.append(m_node)
        mdf = df[df['method'] == method]
        for task in mdf['task_type'].unique():
            t_node = Node(task, 2)
            m_node.children.append(t_node)
            tdf = mdf[mdf['task_type'] == task]
            l3col = 'stat' if task == 'segmentation' else 'aggregation'
            for _, row in tdf.iterrows():
                l3  = row[l3col]
                met = row['metric']
                cov = row['coverage']
                if pd.isna(l3):
                    leaf = Node(met, 3)
                    leaf.coverage = cov
                    leaf.merged   = True
                    t_node.children.append(leaf)
                else:
                    grp = next((c for c in t_node.children
                                if c.name == l3 and not c.merged), None)
                    if grp is None:
                        grp = Node(l3, 3)
                        t_node.children.append(grp)
                    leaf = Node(met, 4)
                    leaf.coverage = cov
                    grp.children.append(leaf)
    return root


# =============================================================================
# CURVED / RADIAL TEXT
# =============================================================================
_FP = FontProperties(family='DejaVu Sans')


@lru_cache(maxsize=None)
def _cw(char, fs):
    if char == ' ':
        return fs * 0.32
    bb = TextPath((0, 0), char, size=fs, prop=_FP).get_extents()
    return bb.width if bb.width > 0 else fs * 0.4


def _make_drawer(pts_per_unit):
    """Return ``(compute_curved_scale, draw_curved,
                 compute_radial_scale, draw_radial)``."""
    ppu = pts_per_unit

    # ── curved (layers 1–3) ───────────────────────────────────────
    def compute_curved_scale(text, radius, a0, a1, fontsize):
        text = text.strip()
        if not text:
            return 1.0
        ws  = [_cw(c, fontsize) for c in text]
        arc = np.deg2rad(abs(a1 - a0)) * radius * ppu
        return min(1.0, arc * 0.85 / max(sum(ws), 1e-6))

    def draw_curved(ax, text, radius, a0, a1, fontsize=10,
                    color='black', fontweight='normal'):
        text = text.strip()
        if not text:
            return
        mid  = (a0 + a1) / 2 % 360
        flip = np.sin(np.deg2rad(mid)) < 0
        fs   = fontsize
        ws   = [_cw(c, fs) for c in text]

        def p2d(p):
            return np.rad2deg(p / (radius * ppu))

        wds  = [p2d(w) for w in ws]
        used = sum(wds)

        if not flip:
            cur = mid + used / 2
            for ch, wd in zip(text, wds):
                ca = cur - wd / 2; cur -= wd
                r  = np.deg2rad(ca)
                ax.text(radius * np.cos(r), radius * np.sin(r), ch,
                        ha='center', va='center', fontsize=fs,
                        rotation=ca - 90, rotation_mode='anchor',
                        color=color, fontweight=fontweight)
        else:
            cur = mid - used / 2
            for ch, wd in zip(text, wds):
                ca = cur + wd / 2; cur += wd
                r  = np.deg2rad(ca)
                ax.text(radius * np.cos(r), radius * np.sin(r), ch,
                        ha='center', va='center', fontsize=fs,
                        rotation=ca + 90, rotation_mode='anchor',
                        color=color, fontweight=fontweight)

    # ── radial (layer 4 / metrics) ────────────────────────────────
    def compute_radial_scale(text, inner, outer, a0, a1, fontsize):
        """Scale ≤ 1 so *text* at *fontsize* fits radially in the wedge."""
        text = text.strip()
        if not text:
            return 1.0
        mid_r = (inner + outer) / 2
        # height constraint: text height (≈ fs) must fit in arc span
        arc_pts    = np.deg2rad(abs(a1 - a0)) * mid_r * ppu
        scale_arc  = arc_pts * 0.75 / max(fontsize, 1e-6)
        # width constraint: text width must fit in radial span
        radial_pts   = (outer - inner) * ppu
        ws           = sum(_cw(c, fontsize) for c in text)
        scale_radial = radial_pts * 0.85 / max(ws, 1e-6)
        return min(1.0, scale_arc, scale_radial)

    def draw_radial(ax, text, inner, outer, a0, a1, fontsize=10,
                    color='black', fontweight='normal'):
        """Draw *text* radially (along the radius), centred in the wedge."""
        text = text.strip()
        if not text:
            return
        mid_angle = (a0 + a1) / 2
        mid_r     = (inner + outer) / 2
        rad = np.deg2rad(mid_angle)
        x, y = mid_r * np.cos(rad), mid_r * np.sin(rad)

        # keep text readable: flip on the left half of the circle
        mid = mid_angle % 360
        if 90 < mid < 270:
            rotation = mid_angle + 180
        else:
            rotation = mid_angle

        ax.text(x, y, text, ha='center', va='center',
                fontsize=fontsize, rotation=rotation,
                rotation_mode='anchor', color=color,
                fontweight=fontweight)

    return compute_curved_scale, draw_curved, compute_radial_scale, draw_radial


# =============================================================================
# COVERAGE COLORBAR
# =============================================================================
def _add_coverage_colorbar(fig, ax, fontsize=24):
    cmap, vmin, vmax = _build_coverage_cmap()
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    sm   = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(
        sm, ax=ax, fraction=0.03, pad=0.04, aspect=30,
        orientation='vertical',
    )
    cbar.set_label('Coverage', fontsize=fontsize)
    ticks = sorted({
        vmin,
        THRESH_UNDER_RED, THRESH_UNDER_ORANGE, THRESH_UNDER_GREEN,
        NOMINAL,
        THRESH_OVER_GREEN, THRESH_OVER_ORANGE, THRESH_OVER_RED,
    })
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f'{t:.0%}' for t in ticks])
    cbar.ax.tick_params(labelsize=fontsize - 2)
    return cbar


# =============================================================================
# DRAW SUNBURST
# =============================================================================
def draw_sunburst(df, fig, ax, center_label=None, center_fontsize=26):
    """
    Draw a single sunburst on the given axes.

    Layers 1–3 use **curved** text (along the arc).
    Layer 4 (metrics) and merged nodes (e.g. MCC) use **radial** text
    (along the radius) so labels can be larger and more readable.
    All nodes of the same type share a uniform font size.
    """
    root = build_tree(df)
    root.compute_avg()
    root.sort_recursive()
    _assign_root_angles(root, 0, 360)

    ax.set_axis_off()
    rmax = INNER_RADIUS + RING_WIDTH * NUM_LAYERS + 0.3
    ax.set(xlim=(-rmax, rmax), ylim=(-rmax, rmax))
    ax.set_aspect('equal')

    fig.canvas.draw()
    p0, p1 = ax.transData.transform([(0, 0), (1, 0)])
    ppu = np.hypot(*(p1 - p0)) * 72.0 / fig.dpi
    (compute_curved_scale, draw_curved,
     compute_radial_scale, draw_radial) = _make_drawer(ppu)

    # -- helpers ---------------------------------------------------
    def _radii(node):
        inner = INNER_RADIUS + RING_WIDTH * (node.layer - 1)
        if node.merged or (node.is_leaf() and node.layer < NUM_LAYERS):
            outer = INNER_RADIUS + RING_WIDTH * NUM_LAYERS
        else:
            outer = inner + RING_WIDTH
        return inner, outer

    def _is_special(node):
        return node.merged or (node.is_leaf() and node.layer < NUM_LAYERS)

    def _uses_radial(node):
        """Layer‑4 nodes and merged nodes → radial text."""
        return node.layer == NUM_LAYERS or node.merged

    # Layer-4 band (for positioning merged‑node text alongside metrics)
    _L4_INNER = INNER_RADIUS + RING_WIDTH * (NUM_LAYERS - 1)
    _L4_OUTER = INNER_RADIUS + RING_WIDTH * NUM_LAYERS

    # ── Pass 1 — per‑layer minimum scale ─────────────────────────
    layer_min_scale  = {}          # layers 1–3, curved
    radial_min_scale = 1.0         # layer 4 + merged, radial

    def _find_scales(node):
        nonlocal radial_min_scale
        if node.layer == 0:
            for c in node.children:
                _find_scales(c)
            return

        if _uses_radial(node):
            ti, to = (_L4_INNER, _L4_OUTER) if node.merged else _radii(node)
            base_fs = LAYER_FONT[NUM_LAYERS]
            sc = compute_radial_scale(
                _pretty(node.name), ti, to,
                node.a0, node.a1, base_fs)
            radial_min_scale = min(radial_min_scale, sc)
        elif not _is_special(node):
            inner, outer = _radii(node)
            mid_r   = (inner + outer) / 2
            base_fs = LAYER_FONT.get(node.layer, 10)
            sc = compute_curved_scale(
                _pretty(node.name), mid_r,
                node.a0, node.a1, base_fs)
            prev = layer_min_scale.get(node.layer, 1.0)
            layer_min_scale[node.layer] = min(prev, sc)

        for c in node.children:
            _find_scales(c)

    _find_scales(root)

    # uniform curved sizes (layers 1–3)
    layer_fs = {
        lay: max(LAYER_FONT[lay] * layer_min_scale.get(lay, 1.0), MIN_FONT)
        for lay in range(1, NUM_LAYERS)
    }
    # uniform radial size (layer 4 + merged)
    radial_fs = max(LAYER_FONT[NUM_LAYERS] * radial_min_scale, MIN_FONT)

    # ── Pass 2 — draw wedges + labels ────────────────────────────
    def _draw(node):
        if node.layer == 0:
            for c in node.children:
                _draw(c)
            return

        inner, outer = _radii(node)
        v  = node.avg_cov if node.avg_cov is not None else 0.5
        fc = coverage_to_color(v)
        tc = _text_color(fc)

        ax.add_patch(mpatches.Wedge(
            (0, 0), outer, node.a0, node.a1,
            width=outer - inner,
            facecolor=fc, edgecolor='white', linewidth=0.6))

        if _uses_radial(node):
            # radial label in the layer-4 band
            ti, to = (_L4_INNER, _L4_OUTER) if node.merged else (inner, outer)
            draw_radial(ax, _pretty(node.name), ti, to,
                        node.a0, node.a1,
                        fontsize=radial_fs, color=tc)
        elif _is_special(node):
            # rare extended leaf — individual curved scale
            mid_r   = (inner + outer) / 2
            base_fs = LAYER_FONT.get(node.layer, 10)
            sc = compute_curved_scale(
                _pretty(node.name), mid_r,
                node.a0, node.a1, base_fs)
            fs = max(base_fs * sc, MIN_FONT)
            draw_curved(ax, _pretty(node.name), mid_r,
                        node.a0, node.a1, fontsize=fs, color=tc)
        else:
            # regular curved text — uniform per‑layer size
            mid_r = (inner + outer) / 2
            draw_curved(ax, _pretty(node.name), mid_r,
                        node.a0, node.a1,
                        fontsize=layer_fs[node.layer], color=tc)

        for c in node.children:
            _draw(c)

    _draw(root)

    if center_label is not None:
        ax.text(0, 0, center_label, ha='center', va='center',
                fontsize=center_fontsize, fontweight='bold', color='#222222')

    _add_coverage_colorbar(fig, ax)