"""The paper's figures of a steering run, drawn from plain arrays.

steered_spectrum_figure  Fig. 3: baseline, each alpha, and real data on log axes, the ratio below, and zooms.
selectivity_figure       R_60 against the target shift, one curve per method (numbers in App. B; no paper figure).
pole_and_width_figure    Fig. 5: fitted pole M and width Gamma of the target against alpha, with the prediction.
save                     writes a figure as PDF.

Each function returns a Figure drawn at the printed width of the paper, in the style of paper.mplstyle, which applies
only inside these functions and `save`. Bin edges, labels, and similar settings are keyword-only; the config holds them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib import colormaps, style
from matplotlib.axes import Axes
from matplotlib.colors import to_hex
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
from numpy.typing import ArrayLike

STYLE_FILE = Path(__file__).with_name("paper.mplstyle")

TEXT_WIDTH = 6.75                           # inches, \textwidth of the paper
COLUMN_WIDTH = 3.25                         # inches, one column
POLE_AND_WIDTH_WIDTH = 0.45 * TEXT_WIDTH    # main.tex includes Fig. 5 at 0.45 \textwidth
PAD = 0.02                                  # inches of white space around the contents of a saved figure

CURVE_LINE_WIDTH = 1.25                     # lines through points
HISTOGRAM_LINE_WIDTH = 0.9                  # step histograms and their ratios
PREDICTION_LINE_WIDTH = 0.8                 # the dotted first-order prediction
MINOR_LOG_TICK_LENGTH = 1.25                # minor ticks of log axes; paper.mplstyle sets 1.5 for linear axes
LOG_LABEL_GAP = 0.25                        # inches, the smallest distance between two labelled powers of ten

BASELINE_COLOR = "black"
REAL_DATA_COLOR = "0.55"
REAL_DATA_OPACITY = 0.6
ALPHA_COLORMAP = "winter"
POINT_COLOR = "darkred"
PREDICTION_COLOR = "0.55"
# One colour per curve of the selectivity figure, in the order of the curves: FKT, Fisher, and difference of means
# in the paper's App. B, then the colours of blocks 0 and 3 in App. D.
CURVE_COLORS = ("darkred", "#1f4e9c", "#e69f00", "#26ad81", "#43a2ca")

SELECTIVITY_KEYS = ("alpha", "target shift", "target shift std", "R_60", "R_60 std")


# ---------------------------------------------------------------- checks of the inputs

def as_masses(masses: ArrayLike, name: str) -> np.ndarray:
    """(N,) float64 copy of a 1-D array of masses in GeV."""
    array = np.asarray(masses, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name}: expected a 1-D array of masses, got shape {array.shape}")
    return array


def as_edges(edges: ArrayLike) -> np.ndarray:
    """(bins + 1,) float64 copy of bin edges in GeV for a log mass axis."""
    array = np.asarray(edges, dtype=np.float64)
    if array.ndim != 1 or len(array) < 2:
        raise ValueError(f"edges: expected a 1-D array of at least two bin edges, got shape {array.shape}")
    if not np.all(np.isfinite(array)) or array[0] <= 0:
        raise ValueError(f"edges: expected finite positive edges in GeV for the log axis, "
                         f"got {array[0]} to {array[-1]}")
    if np.any(np.diff(array) <= 0):
        index = int(np.flatnonzero(np.diff(array) <= 0)[0]) + 1
        raise ValueError(f"edges: expected increasing edges, edge {index} ({array[index]}) is not above edge "
                         f"{index - 1} ({array[index - 1]})")
    return array


def as_columns(arrays: dict[str, ArrayLike], where: str) -> dict[str, np.ndarray]:
    """float64 copies of 1-D arrays of one length with finite values; the keys ending in "std" must be >= 0."""
    columns = {key: np.asarray(value, dtype=np.float64) for key, value in arrays.items()}
    shapes = {key: column.shape for key, column in columns.items()}
    if any(len(shape) != 1 for shape in shapes.values()) or len(set(shapes.values())) != 1:
        raise ValueError(f"{where}: expected 1-D arrays of one length, got shapes {shapes}")
    for key, column in columns.items():
        if not np.all(np.isfinite(column)):
            raise ValueError(f"{where}: {key!r} has non-finite values at {np.flatnonzero(~np.isfinite(column))}")
        if key.endswith("std") and np.any(column < 0):
            raise ValueError(f"{where}: {key!r} has negative standard deviations at {np.flatnonzero(column < 0)}")
    return columns


def check_alphas(alpha: np.ndarray, where: str) -> None:
    """Stops if an alpha appears twice."""
    values, counts = np.unique(alpha, return_counts=True)
    if np.any(counts > 1):
        raise ValueError(f"{where}: alpha {values[counts > 1]} appears more than once")


# ---------------------------------------------------------------- axes and colours

def alpha_colors(alphas: list[float]) -> dict[float, str]:
    """alpha -> colour, equally spaced over the winter colormap in increasing order of alpha."""
    positions = np.linspace(0.0, 1.0, len(alphas))
    return {alpha: to_hex(colormaps[ALPHA_COLORMAP](position)) for alpha, position in zip(sorted(alphas), positions)}


def alpha_label(alpha: float) -> str:
    """Legend entry of one alpha: alpha = +1, alpha = -0.25."""
    return rf"$\alpha = {alpha:+g}$"


def mass_label(pair_label: str) -> str:
    """Axis label of the pair mass, m with the subscript `pair_label` (mathtext), in GeV."""
    return rf"$m_{{{pair_label}}}$ [GeV]"


def log_tick_label(value: float, position: int | None = None) -> str:
    """Label of a power of ten: 1, 10, then 10^n."""
    exponent = int(round(np.log10(value)))
    if exponent == 0:
        return "$1$"
    if exponent == 1:
        return "$10$"
    return rf"$10^{{{exponent}}}$"


def log_axis(axes: Axes, axis_name: str, every: int | None = None) -> None:
    """The paper's log axis: a major tick at every power of ten, minor ticks at 2 to 9 times it, labels 1, 10, 10^n.

    Labels every power of ten if one decade is at least LOG_LABEL_GAP long on the page, else every second power (even
    exponents, so 1 keeps its label), every third, and so on. `every` fixes that step. The step is read from the axis
    when the figure is drawn. Call after the scale is set.
    """
    axis = axes.xaxis if axis_name == "x" else axes.yaxis
    axis.set_major_locator(LogLocator(base=10, numticks=99))
    axis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10), numticks=99))
    axis.set_minor_formatter(NullFormatter())

    def step() -> int:
        if every is not None:
            return every
        low, high = sorted(axis.get_view_interval())
        box = axes.get_window_extent()
        inches = (box.width if axis_name == "x" else box.height) / axes.figure.dpi
        per_decade = inches / max(np.log10(high / low), 1e-9)
        return next(size for size in (1, 2, 3, 4, 5, 10) if size * per_decade >= LOG_LABEL_GAP or size == 10)

    def label(value: float, position: int | None = None) -> str:
        return log_tick_label(value) if int(round(np.log10(value))) % step() == 0 else ""

    axis.set_major_formatter(FuncFormatter(label))
    axes.tick_params(axis=axis_name, which="minor", length=MINOR_LOG_TICK_LENGTH)


def label_minor_ticks_if_few_powers(axes: Axes) -> None:
    """On a log y axis that shows fewer than two powers of ten, label the minor ticks at 3 and 6 times a power of ten,
    as the paper's Upsilon zoom does. Call after `log_axis`."""

    def label(value: float, position: int | None = None) -> str:
        low, high = sorted(axes.yaxis.get_view_interval())
        if np.floor(np.log10(high)) - np.ceil(np.log10(low)) >= 1:       # two or more powers of ten in view
            return ""
        exponent = int(np.floor(np.log10(value) + 1e-9))
        mantissa = int(round(value / 10.0 ** exponent))
        if mantissa not in (3, 6) or not np.isclose(value, mantissa * 10.0 ** exponent):
            return ""
        return rf"${mantissa}\times10^{{{exponent}}}$"

    axes.yaxis.set_minor_formatter(FuncFormatter(label))


def fit_to_width(figure: Figure, width: float) -> None:
    """Widen or narrow `figure`, height kept, until its contents with PAD on each side are `width` inches wide.

    `save` crops the figure to the same contents, so fonts, ticks, and lines print at the sizes of paper.mplstyle
    when the paper includes the PDF at `width`.
    """
    for _ in range(5):
        printed = figure.get_tightbbox().width + 2 * PAD
        if abs(printed - width) < 0.005:
            return
        figure.set_size_inches(figure.get_figwidth() + width - printed, figure.get_figheight())
    raise RuntimeError(f"the figure did not reach a printed width of {width:.3f} in; it is {printed:.3f} in wide")


def save(figure: Figure, path: str | Path) -> None:
    """Write `figure` to `path` as PDF, cropped to its contents with PAD inches around them.

    Matplotlib reads some settings only when it draws, such as the minus sign of tick labels. `save` draws inside the
    paper's style. A figure written with `figure.savefig`, or shown in a notebook, can differ in those settings.
    """
    path = Path(path)
    if path.suffix != ".pdf":
        raise ValueError(f"path: {path} does not end in .pdf; save writes PDF only")
    with style.context(STYLE_FILE):
        figure.savefig(path, format="pdf", bbox_inches="tight", pad_inches=PAD)


# ---------------------------------------------------------------- Fig. 3: the steered spectrum

def counts(masses: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """(bins,) number of masses in each bin."""
    return np.histogram(masses, bins=edges)[0]


def steered_over_baseline(steered_counts: np.ndarray, baseline_counts: np.ndarray) -> np.ndarray:
    """(bins,) steered over baseline counts, NaN where the baseline bin is empty."""
    ratio = np.full(len(baseline_counts), np.nan)
    np.divide(steered_counts, baseline_counts, out=ratio, where=baseline_counts > 0)
    return ratio


def draw_spectra(axes: Axes, edges: np.ndarray, baseline: np.ndarray, steered: dict[float, np.ndarray],
                 real_data: np.ndarray | None, colors: dict[float, str], baseline_label: str | None) -> None:
    """Step histograms in counts: real data grey at the back, one solid line per alpha, and the baseline black and
    dashed in front of every line. With baseline_label None the lines get no legend entries."""
    with_legend = baseline_label is not None
    if real_data is not None:
        axes.stairs(counts(real_data, edges), edges, color=REAL_DATA_COLOR, alpha=REAL_DATA_OPACITY,
                    linewidth=HISTOGRAM_LINE_WIDTH, zorder=2, label="real data" if with_legend else None)
    axes.stairs(counts(baseline, edges), edges, color=BASELINE_COLOR, linewidth=HISTOGRAM_LINE_WIDTH, linestyle="--",
                zorder=10, label=baseline_label)
    for alpha, masses in steered.items():
        axes.stairs(counts(masses, edges), edges, color=colors[alpha], linewidth=HISTOGRAM_LINE_WIDTH, zorder=5,
                    label=alpha_label(alpha) if with_legend else None)


def draw_ratio_panel(axes: Axes, edges: np.ndarray, baseline: np.ndarray, steered: dict[float, np.ndarray],
                     colors: dict[float, str]) -> None:
    """Steered over baseline counts on a log axis from the smallest non-zero ratio, rounded down to a power of ten,
    up to the largest ratio. Bins with a zero ratio, and bins without baseline pairs, fall off the bottom."""
    baseline_counts = counts(baseline, edges)
    ratios = {alpha: steered_over_baseline(counts(masses, edges), baseline_counts) for alpha, masses in steered.items()}
    positive = np.concatenate([ratio[ratio > 0] for ratio in ratios.values()])
    if len(positive) == 0:
        raise ValueError("steered: no bin has pairs in the baseline and in a steered sample, so there is no ratio")
    for alpha, ratio in ratios.items():
        axes.stairs(np.nan_to_num(ratio, nan=0.0), edges, color=colors[alpha], linewidth=HISTOGRAM_LINE_WIDTH,
                    zorder=5)
    axes.axhline(1.0, color=BASELINE_COLOR, linewidth=HISTOGRAM_LINE_WIDTH, linestyle="--", zorder=10)
    axes.set_yscale("log")
    axes.set_ylim(10.0 ** np.floor(np.log10(positive.min())), positive.max())


def steered_spectrum_figure(baseline: ArrayLike, steered: dict[float, ArrayLike], real_data: ArrayLike | None, *,
                            edges: ArrayLike, zooms: dict[str, tuple[float, float]], zoom_bins: int,
                            baseline_label: str, pair_label: str) -> Figure:
    """Fig. 3 of the paper: the pair mass spectrum, the steered over baseline ratio below it, and a row of zooms.

    baseline: (N,) pair masses in GeV generated without the edit. steered: alpha -> (N,) masses of the same N events
    generated with the edit at that alpha; alpha 0 is the baseline and is not allowed. real_data: (N,) real-data masses
    of the same events, or None to draw no real-data line. Every panel shows counts.

    edges: (bins + 1,) increasing positive bin edges in GeV of the spectrum and the ratio panel; the paper uses 200
    log bins from 0.3 to 300 GeV. zooms: label -> (low, high) in GeV, one zoom panel each, left to right; the label is
    mathtext, such as r"$J/\\psi$". zoom_bins: number of equal bins of each zoom, 100 in the paper. baseline_label:
    legend entry of the baseline, r"$\\mathtt{ShellFlow}$" in the paper. pair_label: mathtext subscript of m on the
    mass axes, r"\\mu\\mu" for muons.
    """
    baseline = as_masses(baseline, "baseline")
    if not steered:
        raise ValueError("steered: expected at least one alpha, got none")
    if any(float(alpha) == 0.0 for alpha in steered):
        raise ValueError("steered: alpha 0 is the baseline; pass its masses as `baseline`")
    steered = {alpha: as_masses(steered[alpha], f"steered[{alpha:+g}]") for alpha in sorted(steered)}
    for alpha, masses in steered.items():
        if len(masses) != len(baseline):
            raise ValueError(f"steered[{alpha:+g}]: {len(masses)} masses, the baseline has {len(baseline)}; "
                             f"expected the same events")
    if real_data is not None:
        real_data = as_masses(real_data, "real_data")
        if len(real_data) != len(baseline):
            raise ValueError(f"real_data: {len(real_data)} masses, the baseline has {len(baseline)}; "
                             f"expected the same events")
    edges = as_edges(edges)
    if not zooms:
        raise ValueError("zooms: expected at least one zoom, label -> (low, high) in GeV, got none")
    for label, limits in zooms.items():
        if len(limits) != 2 or not np.all(np.isfinite(limits)) or not limits[0] < limits[1]:
            raise ValueError(f"zooms[{label!r}]: expected (low, high) in GeV with low < high, got {limits}")
    if isinstance(zoom_bins, bool) or not isinstance(zoom_bins, (int, np.integer)) or zoom_bins < 1:
        raise ValueError(f"zoom_bins: expected a positive integer, got {zoom_bins!r}")

    colors = alpha_colors(list(steered))
    with style.context(STYLE_FILE):
        figure = Figure(figsize=(TEXT_WIDTH, TEXT_WIDTH * 8.7 / 12.8))
        grid = figure.add_gridspec(2, len(zooms), height_ratios=[1.75, 1.0], hspace=0.26, wspace=0.28)
        top = grid[0, :].subgridspec(2, 1, height_ratios=[4.0, 1.0], hspace=0.0)
        spectrum = figure.add_subplot(top[0])
        ratio_panel = figure.add_subplot(top[1], sharex=spectrum)

        draw_spectra(spectrum, edges, baseline, steered, real_data, colors, baseline_label)
        spectrum.set_xscale("log")
        spectrum.set_yscale("log")
        spectrum.set_xlim(edges[0], edges[-1])
        spectrum.set_ylabel("Number of events")
        log_axis(spectrum, "x")
        log_axis(spectrum, "y")
        spectrum.tick_params(axis="x", which="both", labelbottom=False)
        spectrum.legend(loc="lower left", ncol=3, handlelength=2.2)

        draw_ratio_panel(ratio_panel, edges, baseline, steered, colors)
        ratio_panel.set_xlim(edges[0], edges[-1])
        log_axis(ratio_panel, "y", every=2)
        ratio_panel.set_ylabel("steered /\nbaseline")
        ratio_panel.set_xlabel(mass_label(pair_label))
        log_axis(ratio_panel, "x")

        for index, (label, (low, high)) in enumerate(zooms.items()):
            zoom = figure.add_subplot(grid[1, index])
            draw_spectra(zoom, np.linspace(low, high, zoom_bins + 1), baseline, steered, real_data, colors, None)
            zoom.set_xlim(low, high)
            zoom.set_yscale("log")
            zoom.set_xlabel(mass_label(pair_label))
            log_axis(zoom, "y")
            label_minor_ticks_if_few_powers(zoom)
            if index == 0:
                zoom.set_ylabel("Number of events")
            zoom.text(0.04, 0.94, label, transform=zoom.transAxes, va="top")
        fit_to_width(figure, TEXT_WIDTH)
    return figure


# ---------------------------------------------------------------- selectivity

def selectivity_figure(curves: dict[str, dict[str, ArrayLike]], *, target_label: str, pair_label: str,
                       max_mass: float) -> Figure:
    """R_60 against the shift of the target: one point per alpha, error bars over seeds, one colour per curve.

    curves: legend label -> the arrays that metrics.py returns, each (K,) over the alphas of the curve: "alpha",
    "target shift" (shift of log m of the target pairs), "target shift std", "R_60", and "R_60 std"; std means the
    standard deviation over seeds. Alpha 0 is the baseline, with shift 0 and R_60 0 by definition, and is not allowed.
    The points of negative alpha and those of positive alpha form two lines of one colour, so that no line crosses
    shift 0, where only the baseline lies.

    target_label: mathtext name of the target on the x axis, such as "$Z$". pair_label: mathtext subscript of m,
    r"\\mu\\mu" for muons. max_mass: the upper mass limit of R_60 in GeV, 60 in the paper, for the y label.
    """
    if not curves:
        raise ValueError("curves: expected at least one curve, got none")
    if len(curves) > len(CURVE_COLORS):
        raise ValueError(f"curves: {len(curves)} curves, the figure has colours for {len(CURVE_COLORS)}")
    columns = {}
    for name, curve in curves.items():
        if set(curve) != set(SELECTIVITY_KEYS):
            raise ValueError(f"curves[{name!r}]: keys {sorted(curve)}, expected exactly {list(SELECTIVITY_KEYS)}")
        columns[name] = as_columns({key: curve[key] for key in SELECTIVITY_KEYS}, f"curves[{name!r}]")
        check_alphas(columns[name]["alpha"], f"curves[{name!r}]")
        if np.any(columns[name]["alpha"] == 0):
            raise ValueError(f"curves[{name!r}]: alpha 0 is the baseline, with shift 0 and R_60 0; leave it out")

    with style.context(STYLE_FILE):
        figure = Figure(figsize=(COLUMN_WIDTH, 2.4))
        axes = figure.subplots()
        for (name, column), color in zip(columns.items(), CURVE_COLORS):
            order = np.argsort(column["alpha"])
            sides = [side for side in (order[column["alpha"][order] < 0], order[column["alpha"][order] > 0])
                     if len(side) > 0]
            for index, side in enumerate(sides):
                axes.errorbar(column["target shift"][side], column["R_60"][side], xerr=column["target shift std"][side],
                              yerr=column["R_60 std"][side], color=color, linestyle="-", marker="o", markersize=2.5,
                              linewidth=CURVE_LINE_WIDTH, capsize=1.2, elinewidth=0.8,
                              label=name if index == 0 else None)
        axes.set_xlabel(rf"{target_label} mass shift, $\Delta\log m_{{{pair_label}}}$")
        axes.set_ylabel(rf"change below {max_mass:g} GeV, $\mathcal{{R}}_{{{max_mass:g}}}$")
        axes.legend(loc="upper center", handlelength=1.6)
        figure.tight_layout()
        fit_to_width(figure, COLUMN_WIDTH)
    return figure


# ---------------------------------------------------------------- Fig. 5: pole and width

def pole_and_width_figure(alpha: ArrayLike, pole: ArrayLike, pole_std: ArrayLike, width: ArrayLike,
                          width_std: ArrayLike, *, first_order_log_slope: float | None) -> Figure:
    """Fig. 5 of the paper: (a) the fitted pole M and (b) the fitted width Gamma of the target against alpha.

    alpha: (K,) steering coefficients; exactly one is 0, the fit of the baseline. pole, width: (K,) fitted values in
    GeV, the mean over seeds. pole_std, width_std: (K,) their standard deviations over seeds in GeV, the error bars.
    first_order_log_slope: S, the sum over steps k of S_C(t_k), which is the first-order slope of log M against alpha
    (0.122 in the paper). With S, a dotted line in each panel shows the value at alpha 0 times exp(alpha S). None
    draws no dotted line.
    """
    columns = as_columns({"alpha": alpha, "pole": pole, "pole std": pole_std, "width": width,
                          "width std": width_std}, "pole_and_width_figure")
    check_alphas(columns["alpha"], "alpha")
    if np.count_nonzero(columns["alpha"] == 0) != 1:
        raise ValueError(f"alpha: expected one alpha 0 for the baseline fit, got alphas {columns['alpha']}")
    if first_order_log_slope is not None and not np.isfinite(first_order_log_slope):
        raise ValueError(f"first_order_log_slope: expected a finite number or None, got {first_order_log_slope}")
    order = np.argsort(columns["alpha"])
    columns = {key: column[order] for key, column in columns.items()}
    alphas = columns["alpha"]

    with style.context(STYLE_FILE):
        figure = Figure(figsize=(POLE_AND_WIDTH_WIDTH, 3.2))
        pole_panel, width_panel = figure.subplots(2, 1, sharex=True)
        for axes, key in ((pole_panel, "pole"), (width_panel, "width")):
            axes.errorbar(alphas, columns[key], yerr=columns[f"{key} std"], color=POINT_COLOR, linestyle="-",
                          marker="o", markersize=2.5, linewidth=CURVE_LINE_WIDTH, capsize=1.2, elinewidth=0.8)
            if first_order_log_slope is not None:
                line = np.linspace(alphas[0], alphas[-1], 121)
                at_zero = columns[key][alphas == 0][0]
                axes.plot(line, at_zero * np.exp(first_order_log_slope * line), color=PREDICTION_COLOR,
                          linewidth=PREDICTION_LINE_WIDTH, linestyle=":")
        pole_panel.set_ylabel(r"pole mass, $M$ [GeV]")
        width_panel.set_ylabel(r"width, $\Gamma$ [GeV]")
        width_panel.set_xlabel(r"coefficient, $\alpha$")
        low, high = width_panel.get_ylim()
        width_panel.set_ylim(low, high + 0.15 * (high - low))       # room for the panel label above the first point
        for axes, letter in ((pole_panel, "a"), (width_panel, "b")):
            axes.text(0.03, 0.95, f"({letter})", transform=axes.transAxes, va="top", ha="left")
        figure.align_ylabels((pole_panel, width_panel))
        figure.tight_layout()
        fit_to_width(figure, POLE_AND_WIDTH_WIDTH)
    return figure
