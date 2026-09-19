"""Small SVG plotting helpers that avoid compiled plotting dependencies."""

from __future__ import annotations

from pathlib import Path
from html import escape

import numpy as np
import pandas as pd


PALETTE = ["#2f6f6d", "#b84a62", "#d89c2b", "#475c9d", "#6f7f4f", "#8d5a99"]


def _svg_header(width: int, height: int) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
        '<rect width="100%" height="100%" fill="#ffffff"/>'
        '<style>text{font-family:Arial,sans-serif;fill:#1f2933} .axis{stroke:#52616b;stroke-width:1}'
        '.grid{stroke:#d9e2ec;stroke-width:1} .title{font-size:18px;font-weight:700}'
        '.label{font-size:12px} .small{font-size:10px}</style>'
    )


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content + "</svg>\n", encoding="utf-8")


def bar_chart(data: pd.DataFrame, category: str, value: str, title: str, output_path: Path) -> None:
    width, height = 920, 520
    margin = dict(left=80, right=30, top=60, bottom=110)
    plot_w = width - margin["left"] - margin["right"]
    plot_h = height - margin["top"] - margin["bottom"]
    labels = data[category].astype(str).tolist()
    values = data[value].astype(float).to_numpy()
    max_v = max(float(np.nanmax(values)) if len(values) else 1.0, 1.0)
    bar_w = plot_w / max(len(values), 1) * 0.72
    gap = plot_w / max(len(values), 1)
    parts = [_svg_header(width, height), f'<text x="30" y="35" class="title">{escape(title)}</text>']
    parts.append(f'<line x1="{margin["left"]}" y1="{margin["top"] + plot_h}" x2="{width - margin["right"]}" y2="{margin["top"] + plot_h}" class="axis"/>')
    parts.append(f'<line x1="{margin["left"]}" y1="{margin["top"]}" x2="{margin["left"]}" y2="{margin["top"] + plot_h}" class="axis"/>')
    for i, (label, val) in enumerate(zip(labels, values)):
        x = margin["left"] + i * gap + (gap - bar_w) / 2
        h = plot_h * (float(val) / max_v)
        y = margin["top"] + plot_h - h
        color = PALETTE[i % len(PALETTE)]
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{color}"/>')
        parts.append(f'<text x="{x + bar_w / 2:.1f}" y="{y - 6:.1f}" text-anchor="middle" class="label">{val:.0f}</text>')
        parts.append(
            f'<text x="{x + bar_w / 2:.1f}" y="{height - 62}" text-anchor="end" transform="rotate(-35 {x + bar_w / 2:.1f},{height - 62})" class="small">{escape(label)}</text>'
        )
    _write(output_path, "".join(parts))


def line_chart(
    data: pd.DataFrame,
    x_col: str,
    y_cols: list[str],
    title: str,
    output_path: Path,
    y_label: str = "metric",
) -> None:
    width, height = 920, 520
    margin = dict(left=80, right=160, top=60, bottom=70)
    plot_w = width - margin["left"] - margin["right"]
    plot_h = height - margin["top"] - margin["bottom"]
    x_values = data[x_col].astype(float).to_numpy()
    if len(x_values) == 0:
        _write(output_path, _svg_header(width, height) + f'<text x="30" y="35" class="title">{escape(title)} (no data)</text>')
        return
    y_values = np.concatenate([data[col].astype(float).replace([np.inf, -np.inf], np.nan).dropna().to_numpy() for col in y_cols])
    y_min = 0.0
    y_max = max(float(np.nanmax(y_values)) if len(y_values) else 1.0, 1.0)
    x_min, x_max = float(np.nanmin(x_values)), float(np.nanmax(x_values))
    if x_min == x_max:
        x_min -= 1
        x_max += 1

    def sx(x: float) -> float:
        return margin["left"] + (x - x_min) / (x_max - x_min) * plot_w

    def sy(y: float) -> float:
        return margin["top"] + plot_h - (y - y_min) / (y_max - y_min) * plot_h

    parts = [_svg_header(width, height), f'<text x="30" y="35" class="title">{escape(title)}</text>']
    for frac in np.linspace(0, 1, 6):
        y = margin["top"] + frac * plot_h
        parts.append(f'<line x1="{margin["left"]}" y1="{y:.1f}" x2="{width - margin["right"]}" y2="{y:.1f}" class="grid"/>')
    parts.append(f'<line x1="{margin["left"]}" y1="{margin["top"] + plot_h}" x2="{width - margin["right"]}" y2="{margin["top"] + plot_h}" class="axis"/>')
    parts.append(f'<line x1="{margin["left"]}" y1="{margin["top"]}" x2="{margin["left"]}" y2="{margin["top"] + plot_h}" class="axis"/>')
    parts.append(f'<text x="22" y="{margin["top"] + plot_h / 2:.1f}" class="label" transform="rotate(-90 22,{margin["top"] + plot_h / 2:.1f})">{escape(y_label)}</text>')
    for idx, col in enumerate(y_cols):
        color = PALETTE[idx % len(PALETTE)]
        points = []
        for _, row in data[[x_col, col]].dropna().iterrows():
            points.append(f'{sx(float(row[x_col])):.1f},{sy(float(row[col])):.1f}')
        if points:
            parts.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2.5"/>')
            for point in points:
                x, y = point.split(",")
                parts.append(f'<circle cx="{x}" cy="{y}" r="3.5" fill="{color}"/>')
        legend_y = margin["top"] + idx * 22
        parts.append(f'<rect x="{width - 140}" y="{legend_y - 11}" width="12" height="12" fill="{color}"/>')
        parts.append(f'<text x="{width - 122}" y="{legend_y}" class="label">{escape(col)}</text>')
    for x in sorted(set(x_values.tolist())):
        parts.append(f'<text x="{sx(x):.1f}" y="{height - 35}" text-anchor="middle" class="label">{x:g}</text>')
    _write(output_path, "".join(parts))


def confusion_matrix_svg(cm: dict[str, int], title: str, output_path: Path) -> None:
    width, height = 520, 430
    parts = [_svg_header(width, height), f'<text x="30" y="35" class="title">{escape(title)}</text>']
    labels = [("TN", cm.get("tn", 0)), ("FP", cm.get("fp", 0)), ("FN", cm.get("fn", 0)), ("TP", cm.get("tp", 0))]
    max_v = max([v for _, v in labels] + [1])
    positions = [(90, 90), (270, 90), (90, 240), (270, 240)]
    for (label, val), (x, y) in zip(labels, positions):
        opacity = 0.25 + 0.65 * val / max_v
        parts.append(f'<rect x="{x}" y="{y}" width="150" height="120" fill="#2f6f6d" opacity="{opacity:.2f}"/>')
        parts.append(f'<text x="{x + 75}" y="{y + 48}" text-anchor="middle" class="label">{label}</text>')
        parts.append(f'<text x="{x + 75}" y="{y + 82}" text-anchor="middle" style="font-size:28px;font-weight:700">{val}</text>')
    parts.append('<text x="255" y="390" text-anchor="middle" class="label">Predicted classes vs actual classes</text>')
    _write(output_path, "".join(parts))


def scatter_line_chart(
    data: pd.DataFrame,
    x_col: str,
    y_prob_col: str,
    y_actual_col: str,
    title: str,
    output_path: Path,
    max_points: int = 240,
) -> None:
    sample = data.sort_values(x_col).tail(max_points).copy()
    sample["_x"] = np.arange(len(sample))
    line_chart(sample, "_x", [y_prob_col, y_actual_col], title, output_path, y_label="probability / actual")
