"""Small SVG chart builders. Colours and fonts come from static/style.css."""
from markupsafe import Markup, escape


def _fmt(value, span):
    if span >= 50:
        return f"{value:.0f}"
    if span >= 5:
        return f"{value:.1f}"
    if span >= 0.5:
        return f"{value:.2f}"
    return f"{value:.3f}"


def line_chart(xs, series, width=640, height=250, y_min=None, y_max=None,
               x_label="Cycle", y_label="", y_ticks=4):
    """series: list of {"name": str, "ys": list, "cls": "line-main" | "line-ref"}"""
    left, right, top, bottom = 54, 14, 22, 36
    pw, ph = width - left - right, height - top - bottom

    xs = [float(x) for x in xs]
    all_y = [float(y) for s in series for y in s["ys"]]
    lo = min(all_y) if y_min is None else y_min
    hi = max(all_y) if y_max is None else y_max
    if hi == lo:
        hi = lo + 1
    if y_min is None and y_max is None:
        pad = (hi - lo) * 0.06
        lo, hi = lo - pad, hi + pad
    x0, x1 = min(xs), max(xs)
    if x1 == x0:
        x1 = x0 + 1

    def sx(x):
        return left + (x - x0) / (x1 - x0) * pw

    def sy(y):
        return top + (1 - (y - lo) / (hi - lo)) * ph

    out = [f'<svg class="chart" viewBox="0 0 {width} {height}" role="img">']
    if y_label:
        out.append(f'<text class="axis-text" x="{left}" y="11">{escape(y_label)}</text>')

    for i in range(y_ticks + 1):                          # horizontal grid + y labels
        v = lo + (hi - lo) * i / y_ticks
        y = sy(v)
        out.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}"/>')
        out.append(f'<text class="axis-text" x="{left - 8}" y="{y + 4:.1f}" text-anchor="end">{_fmt(v, hi - lo)}</text>')

    for i in range(5):                                    # x labels
        v = x0 + (x1 - x0) * i / 4
        out.append(f'<text class="axis-text" x="{sx(v):.1f}" y="{height - 16}" text-anchor="middle">{v:.0f}</text>')
    out.append(f'<text class="axis-text" x="{left + pw / 2:.1f}" y="{height - 2}" text-anchor="middle">{escape(x_label)}</text>')
    out.append(f'<line class="axis" x1="{left}" y1="{top + ph}" x2="{width - right}" y2="{top + ph}"/>')

    for s in series:
        pts = " ".join(f"{sx(x):.1f},{sy(float(y)):.1f}" for x, y in zip(xs, s["ys"]))
        out.append(f'<polyline class="{s.get("cls", "line-main")}" points="{pts}"/>')

    if len(series) > 1:                                   # legend inside the plot, top right
        lx = width - right - 8
        for i, s in enumerate(reversed(series)):
            ly = top + 12 + i * 16
            out.append(f'<text class="legend-text" x="{lx - 26}" y="{ly + 4}" text-anchor="end">{escape(s["name"])}</text>')
            out.append(f'<line class="{s.get("cls", "line-main")}" x1="{lx - 20}" y1="{ly}" x2="{lx}" y2="{ly}"/>')

    out.append("</svg>")
    return Markup("".join(out))


def bar_chart(items, best=None, width=560, decimals=2):
    """items: list of (label, value). The bar whose label == best is highlighted."""
    label_w, value_w, row_h = 118, 64, 36
    vmax = max(v for _, v in items) or 1
    pw = width - label_w - value_w
    height = row_h * len(items) + 6

    out = [f'<svg class="chart" viewBox="0 0 {width} {height}" role="img">']
    for i, (label, value) in enumerate(items):
        y = 3 + i * row_h
        w = value / vmax * pw
        cls = "bar bar-best" if label == best else "bar"
        out.append(f'<text class="bar-label" x="0" y="{y + row_h / 2 + 4:.1f}">{escape(label)}</text>')
        out.append(f'<rect class="{cls}" x="{label_w}" y="{y + 7}" width="{w:.1f}" height="{row_h - 14}"/>')
        out.append(f'<text class="bar-value" x="{label_w + w + 8:.1f}" y="{y + row_h / 2 + 4:.1f}">{value:.{decimals}f}</text>')
    out.append("</svg>")
    return Markup("".join(out))
