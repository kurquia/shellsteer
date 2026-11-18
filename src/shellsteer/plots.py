"""The paper's figures of a steering run, drawn from plain arrays.

steered_spectrum_figure  Fig. 3: baseline, each alpha, and real data on log axes, the ratio below, and zooms.
selectivity_figure       R_60 against the target shift, one curve per method (numbers in App. B; no paper figure).
pole_and_width_figure    Fig. 5: fitted pole M and width Gamma of the target against alpha, with the prediction.
networks_pole_and_width_figure  Fig. 13 (App. C): the same for several runs, such as the three networks, in one figure.
ablation_figure          Fig. 7: baseline, mean ablation, and real data on log axes, an inset of the target peak, the
                         ratio below.
kinematics_figure        Fig. 4: p_T, eta, and phi of the edited particle type, each with its ratio panel.
all_particles_figure     Fig. 14: p_T, eta, phi, E, and m of every other particle type, one row per type.
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
from matplotlib.patches import ConnectionPatch, Rectangle
from matplotlib.lines import Line2D
from matplotlib.ticker import (FixedLocator, FuncFormatter, LogLocator, MaxNLocator, MultipleLocator, NullFormatter,
                               NullLocator)
from numpy.typing import ArrayLike

from shellsteer.kinematics import LOG_VARIABLES, MASS, VARIABLES, Panel, steered_over_baseline as ratio_of_counts

STYLE_FILE = Path(__file__).with_name("paper.mplstyle")

TEXT_WIDTH = 6.75                           # inches, \textwidth of the paper
COLUMN_WIDTH = 3.25                         # inches, one column
POLE_AND_WIDTH_WIDTH = 0.45 * TEXT_WIDTH    # main.tex includes Fig. 5 at 0.45 \textwidth
NETWORKS_POLE_AND_WIDTH_HEIGHT = 2.6        # inches, Fig. 13 before cropping, drawn at POLE_AND_WIDTH_WIDTH
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
# One colour per run of Fig. 13 (App. C), in legend order: ShellFlow, ShellFlow-B, ShellGATr.
NETWORK_COLORS = ("#D55E00", "#26ad81", "#43a2ca")

SELECTIVITY_KEYS = ("alpha", "target shift", "target shift std", "R_60", "R_60 std")

ABLATION_COLOR = to_hex(colormaps[ALPHA_COLORMAP](0.0))   # the blue end of the alpha colours
ABLATION_LABEL = "Ablation"
ABLATION_HEIGHT = COLUMN_WIDTH * 0.78 * 1.175              # inches, Fig. 7 before cropping
INSET_POSITION = (0.40, 0.16, 0.33, 0.27)                  # left, bottom, width, height in the spectrum's axes
INSET_FONT_SIZE = 5.5                                      # tick labels and x label of the inset
INSET_TARGET_SIZE = 7                                      # the target's name inside the inset
RATIO_LABEL_SIZE = 6                                       # the two-line y label of the ratio panel of Fig. 7
LEGEND_ROOM = 0.24                                         # top of the log y axis of Fig. 7 left free for the legend
BOX_COLOR = "0.3"                                          # the box around the inset's range and its two lines

# Figs. 4 and 14: the kinematics of every particle type.
KINEMATICS_HEIGHT = 2.6                     # inches, Fig. 4 before cropping
KINEMATICS_LABELS = {"pt": r"$p_T$ [GeV]", "eta": r"$\eta$", "phi": r"$\phi$", "E": r"$E$ [GeV]", "m": r"$m$ [GeV]"}
MASS_X_MARGIN = 0.04                        # Fig. 14, m: room on each side of the bins, as a fraction of their range
# ShellFlow type code -> name inside the panels of Fig. 4, and title of the row of Fig. 14 (bold through mathtext,
# because cmr10 has no bold weight).
PARTICLE_LABELS = {
    13: ("muons", r"$\mathbf{Muon\ kinematics}$"),
    11: ("electrons", r"$\mathbf{Electron\ kinematics}$"),
    22: ("photons", r"$\mathbf{Photon\ kinematics}$"),
    1: ("jets", r"$\mathbf{Jet\ kinematics}$"),
    10: ("large-R jets", r"$\mathbf{Large}$-$\mathbf{R\ jet\ kinematics}$"),
    15: (r"hadronic $\tau$", r"$\mathbf{Hadronic}\ \tau\ \mathbf{kinematics}$")}
KINEMATICS_RATIO_LINE_WIDTH = 0.7           # the dashed line at 1 of the ratio panels
KINEMATICS_LEGEND_SPACING = 2.2             # space between legend entries, in font sizes
SHORT_LOG_RANGE = 30                        # Fig. 4: a log x axis shorter than this factor also labels 2 and 5 x 10^n
TWO_DECADES = 100                           # Fig. 14: a log x axis shorter than this factor starts below its decade
ROW_TITLE_SIZE = 9                          # Fig. 14: the title of each row
# Fig. 14, in inches from the top: legend, then per row title, gap, panels, tick labels; x titles once at the bottom.
ROW_LEGEND, ROW_TITLE, ROW_GAP, ROW_PANELS, ROW_TICKS, ROW_FOOT = 0.22, 0.14, 0.10, 0.65, 0.24, 0.17


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

def pole_and_width_columns(alpha: ArrayLike, pole: ArrayLike, pole_std: ArrayLike, width: ArrayLike,
                           width_std: ArrayLike, first_order_log_slope: float | None, where: str
                           ) -> dict[str, np.ndarray]:
    """The checked columns of one series of pole and width against alpha, sorted by alpha; stops on a repeated alpha,
    on a series without exactly one alpha 0, or on a non-finite slope."""
    columns = as_columns({"alpha": alpha, "pole": pole, "pole std": pole_std, "width": width,
                          "width std": width_std}, where)
    check_alphas(columns["alpha"], where)
    if np.count_nonzero(columns["alpha"] == 0) != 1:
        raise ValueError(f"{where}: expected one alpha 0 for the baseline fit, got alphas {columns['alpha']}")
    if first_order_log_slope is not None and not np.isfinite(first_order_log_slope):
        raise ValueError(f"{where}: first_order_log_slope: expected a finite number or None, got "
                         f"{first_order_log_slope}")
    order = np.argsort(columns["alpha"])
    return {key: column[order] for key, column in columns.items()}


def draw_pole_and_width(pole_panel: Axes, width_panel: Axes, columns: dict[str, np.ndarray],
                        first_order_log_slope: float | None, *, color: str, prediction_color: str,
                        label: str | None = None, zorder: float | None = None) -> None:
    """One series in both panels: the fitted values with their seed spread, and, with S, the dotted value at alpha 0
    times exp(alpha S) from the smallest to the largest alpha."""
    alphas = columns["alpha"]
    order = {} if zorder is None else {"zorder": zorder}
    for axes, key in ((pole_panel, "pole"), (width_panel, "width")):
        axes.errorbar(alphas, columns[key], yerr=columns[f"{key} std"], color=color, linestyle="-", marker="o",
                      markersize=2.5, linewidth=CURVE_LINE_WIDTH, capsize=1.2, elinewidth=0.8, label=label, **order)
        if first_order_log_slope is not None:
            line = np.linspace(alphas[0], alphas[-1], 121)
            at_zero = columns[key][alphas == 0][0]
            axes.plot(line, at_zero * np.exp(first_order_log_slope * line), color=prediction_color,
                      linewidth=PREDICTION_LINE_WIDTH, linestyle=":", **({} if zorder is None else {"zorder": 3}))


def finish_pole_and_width(figure: Figure, pole_panel: Axes, width_panel: Axes) -> None:
    """Axis labels, room above the width panel for its label, the panel labels (a) and (b), and the layout."""
    pole_panel.set_ylabel(r"pole mass, $M$ [GeV]")
    width_panel.set_ylabel(r"width, $\Gamma$ [GeV]")
    width_panel.set_xlabel(r"coefficient, $\alpha$")
    low, high = width_panel.get_ylim()
    width_panel.set_ylim(low, high + 0.15 * (high - low))       # room for the panel label above the first point
    for axes, letter in ((pole_panel, "a"), (width_panel, "b")):
        axes.text(0.03, 0.95, f"({letter})", transform=axes.transAxes, va="top", ha="left")
    figure.align_ylabels((pole_panel, width_panel))
    figure.tight_layout()


def pole_and_width_figure(alpha: ArrayLike, pole: ArrayLike, pole_std: ArrayLike, width: ArrayLike,
                          width_std: ArrayLike, *, first_order_log_slope: float | None) -> Figure:
    """Fig. 5 of the paper: (a) the fitted pole M and (b) the fitted width Gamma of the target against alpha.

    alpha: (K,) steering coefficients; exactly one is 0, the fit of the baseline. pole, width: (K,) fitted values in
    GeV, the mean over seeds. pole_std, width_std: (K,) their standard deviations over seeds in GeV, the error bars.
    first_order_log_slope: S, the sum over steps k of S_C(t_k), which is the first-order slope of log M against alpha
    (0.122 in the paper). With S, a dotted line in each panel shows the value at alpha 0 times exp(alpha S). None
    draws no dotted line.
    """
    columns = pole_and_width_columns(alpha, pole, pole_std, width, width_std, first_order_log_slope,
                                     "pole_and_width_figure")
    with style.context(STYLE_FILE):
        figure = Figure(figsize=(POLE_AND_WIDTH_WIDTH, 3.2))
        pole_panel, width_panel = figure.subplots(2, 1, sharex=True)
        draw_pole_and_width(pole_panel, width_panel, columns, first_order_log_slope, color=POINT_COLOR,
                            prediction_color=PREDICTION_COLOR)
        finish_pole_and_width(figure, pole_panel, width_panel)
        fit_to_width(figure, POLE_AND_WIDTH_WIDTH)
    return figure


def networks_pole_and_width_figure(series: dict[str, dict[str, ArrayLike]],
                                   first_order_log_slopes: dict[str, float | None]) -> Figure:
    """Fig. 13 of the paper (App. C): pole M and width Gamma against alpha of several runs, such as the Z vector at
    block 1 of the three networks, in one figure.

    series: legend label -> {"alpha", "pole", "pole std", "width", "width std"}, each run as `pole_and_width_figure`
    takes it, in legend order. The colours are NETWORK_COLORS in that order, and the first series is drawn in front.
    first_order_log_slopes: label -> S of the run, or None; with S, a dotted line in the series' colour shows its value
    at alpha 0 times exp(alpha S).
    """
    if not 1 <= len(series) <= len(NETWORK_COLORS):
        raise ValueError(f"series: expected 1 to {len(NETWORK_COLORS)} runs, got {len(series)}")
    if set(first_order_log_slopes) != set(series):
        raise ValueError(f"first_order_log_slopes: expected the labels {list(series)}, got {list(first_order_log_slopes)}")
    columns = {label: pole_and_width_columns(**{key.replace(" ", "_"): value for key, value in data.items()},
                                             first_order_log_slope=first_order_log_slopes[label], where=label)
               for label, data in series.items()}
    with style.context(STYLE_FILE):
        figure = Figure(figsize=(POLE_AND_WIDTH_WIDTH, NETWORKS_POLE_AND_WIDTH_HEIGHT))
        pole_panel, width_panel = figure.subplots(2, 1, sharex=True)
        drawn = list(zip(columns, NETWORK_COLORS))
        for rank, (label, color) in [*enumerate(drawn)][1:] + [(0, drawn[0])]:     # the first series last, in front
            draw_pole_and_width(pole_panel, width_panel, columns[label], first_order_log_slopes[label], color=color,
                                prediction_color=color, label=label, zorder=6 if rank == 0 else 5)
        handles, labels = pole_panel.get_legend_handles_labels()
        order = [labels.index(label) for label in columns]
        pole_panel.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper left",
                          bbox_to_anchor=(0.0, 0.87), fontsize=6.5, handlelength=1.6, frameon=False)
        finish_pole_and_width(figure, pole_panel, width_panel)
        fit_to_width(figure, POLE_AND_WIDTH_WIDTH)
    return figure


# ---------------------------------------------------------------- Fig. 7: the mean ablation

def as_seeds(masses: list[ArrayLike], name: str) -> list[np.ndarray]:
    """1-D float64 copies of the masses of each seed; stops on an empty list."""
    if not masses:
        raise ValueError(f"{name}: expected the masses of at least one seed, got none")
    return [as_masses(m, f"{name}[{index}]") for index, m in enumerate(masses)]


def draw_ablation(axes: Axes, edges: np.ndarray, baseline: np.ndarray, ablated: np.ndarray,
                  real_data: np.ndarray | None, baseline_label: str | None) -> None:
    """Step histograms of counts per bin: real data grey at the back, the ablation solid, and the baseline black and
    dashed in front. With baseline_label None the lines get no legend entries."""
    with_legend = baseline_label is not None
    if real_data is not None:
        axes.stairs(real_data, edges, color=REAL_DATA_COLOR, alpha=REAL_DATA_OPACITY, linewidth=HISTOGRAM_LINE_WIDTH,
                    zorder=2, label="real data" if with_legend else None)
    axes.stairs(baseline, edges, color=BASELINE_COLOR, linewidth=HISTOGRAM_LINE_WIDTH, linestyle="--", zorder=10,
                label=baseline_label)
    axes.stairs(ablated, edges, color=ABLATION_COLOR, linewidth=HISTOGRAM_LINE_WIDTH, zorder=5,
                label=ABLATION_LABEL if with_legend else None)


def ablation_figure(baseline: list[ArrayLike], ablated: list[ArrayLike], real_data: ArrayLike | None, *,
                    edges: ArrayLike, inset_edges: ArrayLike, inset_range: tuple[float, float],
                    ratio_range: tuple[float, float], baseline_label: str, pair_label: str,
                    target_label: str) -> Figure:
    """Fig. 7 of the paper: the pair mass spectrum without and with the mean ablation (Eq. 24), an inset of the target
    peak on linear axes, and the ratio of baseline over ablation below.

    baseline, ablated: one (N_s,) array of pair masses in GeV per seed, the same events within a seed. Each line is
    the mean over seeds of the counts per bin. real_data: (N,) real-data masses, counted once, or None to draw no
    real-data line. Bins without ablated pairs have no ratio.

    edges: (bins + 1,) increasing positive bin edges in GeV of the spectrum and the ratio; the paper uses 200 log
    bins from 0.3 to 300 GeV. inset_edges: bin edges of the inset, 100 equal bins from 50 to 140 GeV in the paper.
    inset_range: (low, high) in GeV that the inset shows, inside inset_edges; a box marks it on the spectrum.
    ratio_range: (low, high) of the linear y axis of the ratio. baseline_label: legend entry of the baseline.
    pair_label: mathtext subscript of m on the mass axes. target_label: mathtext name of the target, shown in the inset.
    """
    baseline, ablated = as_seeds(baseline, "baseline"), as_seeds(ablated, "ablated")
    if len(baseline) != len(ablated):
        raise ValueError(f"ablated: {len(ablated)} seeds, baseline has {len(baseline)}; expected one array per seed")
    for index, (base, abl) in enumerate(zip(baseline, ablated)):
        if len(base) != len(abl):
            raise ValueError(f"ablated[{index}]: {len(abl)} masses, baseline[{index}] has {len(base)}; expected the "
                             f"same events")
    if real_data is not None:
        real_data = as_masses(real_data, "real_data")
    edges, inset_edges = as_edges(edges), as_edges(inset_edges)
    low, high = inset_range
    if not inset_edges[0] <= low < high <= inset_edges[-1]:
        raise ValueError(f"inset_range: expected (low, high) with low < high inside the inset edges "
                         f"{inset_edges[0]:g} to {inset_edges[-1]:g} GeV, got {inset_range}")
    if not 0 <= ratio_range[0] < ratio_range[1]:
        raise ValueError(f"ratio_range: expected (low, high) with 0 <= low < high, got {ratio_range}")

    def histograms(bins: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        mean = [np.mean([counts(m, bins) for m in per_seed], axis=0) for per_seed in (baseline, ablated)]
        return mean[0], mean[1], None if real_data is None else counts(real_data, bins)

    spectra, inset_spectra = histograms(edges), histograms(inset_edges)
    drawn = [h for h in spectra if h is not None]
    in_box = (edges[:-1] >= low) & (edges[1:] <= high)
    box_counts = np.concatenate([h[in_box] for h in drawn])
    if not np.any(box_counts > 0):
        raise ValueError(f"inset_range: no spectrum bin inside {low:g} to {high:g} GeV has pairs, so the box around "
                         f"the inset's range has no height")

    with style.context(STYLE_FILE):
        figure = Figure(figsize=(COLUMN_WIDTH, ABLATION_HEIGHT))
        grid = figure.add_gridspec(2, 1, height_ratios=[4.0, 0.7], hspace=0.0)
        spectrum = figure.add_subplot(grid[0])
        ratio_panel = figure.add_subplot(grid[1], sharex=spectrum)

        draw_ablation(spectrum, edges, *spectra, baseline_label)
        spectrum.set_xscale("log")
        spectrum.set_yscale("log")
        spectrum.set_xlim(edges[0], edges[-1])
        log_axis(spectrum, "x")
        log_axis(spectrum, "y")
        spectrum.set_ylabel("Number of events")
        spectrum.tick_params(axis="x", which="both", labelbottom=False)
        bottom, top = np.log10(spectrum.get_ylim()[0]), np.log10(max(h.max() for h in drawn))
        spectrum.set_ylim(10.0 ** bottom, 10.0 ** (bottom + (top - bottom) / (1 - LEGEND_ROOM)))
        spectrum.legend(loc="upper right")

        inset = spectrum.inset_axes(INSET_POSITION)
        inset.set_zorder(20)
        inset.set_facecolor("white")
        draw_ablation(inset, inset_edges, *inset_spectra, None)
        inset.set_xlim(low, high)
        inset.set_ylim(0, None)
        inset.tick_params(which="both", labelsize=INSET_FONT_SIZE)
        inset.set_xlabel(mass_label(pair_label), fontsize=INSET_FONT_SIZE, labelpad=1)
        inset.text(0.06, 0.92, target_label, transform=inset.transAxes, va="top", fontsize=INSET_TARGET_SIZE)
        box_low, box_high = 0.7 * box_counts[box_counts > 0].min(), 1.4 * box_counts.max()
        spectrum.add_patch(Rectangle((low, box_low), high - low, box_high - box_low, fill=False, edgecolor=BOX_COLOR,
                                     linewidth=1.0, zorder=15))
        for box_y, inset_y in ((box_low, 0.0), (box_high, 1.0)):
            figure.add_artist(ConnectionPatch((low, box_y), (1.0, inset_y), coordsA=spectrum.transData,
                                              coordsB=inset.transAxes, color=BOX_COLOR, linewidth=0.8, zorder=15))

        ratio = steered_over_baseline(spectra[0], spectra[1])        # baseline over ablation, NaN without ablated pairs
        ratio_panel.stairs(ratio, edges, color=ABLATION_COLOR, linewidth=HISTOGRAM_LINE_WIDTH, zorder=5, baseline=None)
        ratio_panel.axhline(1.0, color=BASELINE_COLOR, linewidth=HISTOGRAM_LINE_WIDTH, linestyle="--", zorder=10)
        ratio_panel.set_ylim(*ratio_range)
        ratio_panel.yaxis.set_major_locator(MultipleLocator(1.0))
        ratio_panel.yaxis.set_minor_locator(MultipleLocator(0.5))
        ratio_panel.set_xlim(edges[0], edges[-1])
        log_axis(ratio_panel, "x")
        ratio_panel.set_ylabel(f"{baseline_label} /\n{ABLATION_LABEL}", fontsize=RATIO_LABEL_SIZE)
        ratio_panel.set_xlabel(mass_label(pair_label))
        fit_to_width(figure, COLUMN_WIDTH)
    return figure


# ---------------------------------------------------------------- Figs. 4 and 14: kinematics of every particle type

def ratio_axis_range(ratios: list[np.ndarray]) -> tuple[float, float]:
    """y range of a ratio panel: 1 +- the largest deviation, rounded up, when every ratio lies within 1 +- 0.25; else
    0 to the largest ratio rounded up to 0.1, at most 2."""
    values = np.concatenate([ratio[np.isfinite(ratio)] for ratio in ratios] + [np.array([1.0])])
    deviation = np.max(np.abs(values - 1.0))
    if deviation < 0.25:
        step = next(step for step in (0.001, 0.002, 0.005, 0.01, 0.02, 0.05) if deviation * 1.1 <= 10 * step)
        half = max(step * np.ceil(deviation * 1.1 / step - 1e-9), 0.005)
        return 1.0 - half, 1.0 + half
    return 0.0, float(min(2.0, max(1.05, 0.1 * np.ceil(values.max() * 1.05 / 0.1 - 1e-9))))


def set_ratio_ticks(axes: Axes, low: float, high: float) -> None:
    """Three ticks away from the edges of a ratio panel; the one at 1 has no label, the dashed line marks it."""
    if low == 0.0:
        ticks = [tick for tick in (0.5, 1.0, 1.5) if tick < 0.85 * high]
    else:
        step = max([s for s in (0.001, 0.002, 0.005, 0.01, 0.02, 0.025, 0.05, 0.1, 0.2) if s <= 0.75 * (high - 1)]
                   or [0.001])
        ticks = [1.0 - step, 1.0, 1.0 + step]
    axes.yaxis.set_major_locator(FixedLocator(ticks))
    axes.yaxis.set_major_formatter(FuncFormatter(lambda value, position: "" if np.isclose(value, 1.0)
                                                 else f"${value:g}$"))
    axes.yaxis.set_minor_locator(NullLocator())


def check_panel(panel: Panel, where: str) -> None:
    """Stops unless the counts of `panel` match its edges and the steered alphas exclude 0."""
    bins = len(panel.edges) - 1
    if np.ndim(panel.edges) != 1 or bins < 1 or np.any(np.diff(panel.edges) <= 0):
        raise ValueError(f"{where}: expected increasing bin edges, got shape {np.shape(panel.edges)}")
    if not panel.steered or any(float(alpha) == 0.0 for alpha in panel.steered):
        raise ValueError(f"{where}: expected at least one steered alpha other than 0, got {list(panel.steered)}")
    for name, values in (("real", panel.real), ("baseline", panel.baseline),
                         *((f"steered[{alpha:+g}]", values) for alpha, values in panel.steered.items())):
        if np.shape(values) != (bins,):
            raise ValueError(f"{where}: {name} has shape {np.shape(values)}, the edges give {bins} bins")


def draw_kinematics_panel(axes: Axes, ratio_axes: Axes, variable: str, panel: Panel, colors: dict[float, str], *,
                          min_baseline: int, x_margin: float, top_factor: float, start_below_decade: bool,
                          ratio_range: tuple[float, float] | None) -> tuple[float, float]:
    """One panel of Figs. 4 and 14 and its ratio panel; returns the y range of the ratio panel.

    Counts: real data grey at the back, one solid line per alpha, the baseline black and dashed in front; log axes for
    p_T and E, linear x with MASS_X_MARGIN and log y for m, linear y from 0 to top_factor x the highest bin for eta and
    phi. Ratio: steered over baseline in the bins with at least `min_baseline` baseline particles, a gap elsewhere, on
    `ratio_range`, or on the range of `ratio_axis_range` for None. x_margin: room as a factor on log x axes,
    beyond the first and last bin with real or baseline particles. A log x axis shorter than two decades starts
    below its first power of ten with start_below_decade (Fig. 14), so that a power of ten is labelled; without it
    (Fig. 4), an axis shorter than SHORT_LOG_RANGE labels 2 and 5 times a power of ten under the ratio panel.
    """
    edges = np.asarray(panel.edges, dtype=np.float64)
    real, base = np.asarray(panel.real, dtype=np.float64), np.asarray(panel.baseline, dtype=np.float64)
    axes.stairs(real, edges, color=REAL_DATA_COLOR, alpha=REAL_DATA_OPACITY, linewidth=HISTOGRAM_LINE_WIDTH, zorder=2)
    axes.stairs(base, edges, color=BASELINE_COLOR, linewidth=HISTOGRAM_LINE_WIDTH, linestyle="--", zorder=10)
    ratios = []
    for alpha, steered in panel.steered.items():
        axes.stairs(steered, edges, color=colors[alpha], linewidth=HISTOGRAM_LINE_WIDTH, zorder=5)
        ratio = ratio_of_counts(np.asarray(steered, dtype=np.float64), base, min_baseline)
        ratio_axes.stairs(ratio, edges, color=colors[alpha], linewidth=HISTOGRAM_LINE_WIDTH, zorder=5, baseline=None)
        ratios.append(ratio)
    ratio_axes.axhline(1.0, color=BASELINE_COLOR, linewidth=KINEMATICS_RATIO_LINE_WIDTH, linestyle="--", zorder=10)
    if variable in LOG_VARIABLES:
        filled = np.flatnonzero(real + base > 0)
        if len(filled) == 0:
            raise ValueError(f"{variable}: no bin has real or baseline particles")
        low, high = edges[filled[0]] / x_margin, edges[filled[-1] + 1] * x_margin
        if start_below_decade and high / low < TWO_DECADES:
            low = min(low, 0.9 * 10 ** np.floor(np.log10(low)))
        for each in (axes, ratio_axes):
            each.set_xscale("log")
            log_axis(each, "x")
        axes.set_xlim(low, high)
        if not start_below_decade and high / low < SHORT_LOG_RANGE:
            ratio_axes.xaxis.set_minor_formatter(FuncFormatter(
                lambda value, position: f"${value:g}$" if round(value / 10 ** np.floor(np.log10(value)), 6) in (2, 5)
                else ""))
        axes.set_yscale("log")
        log_axis(axes, "y")
        axes.set_ylim(0.5, max(base.max(), real.max(), 1.0) * 8)
    elif variable == MASS:
        margin = MASS_X_MARGIN * (edges[-1] - edges[0])
        axes.set_xlim(edges[0] - margin, edges[-1] + margin)
        axes.set_yscale("log")
        log_axis(axes, "y")
        axes.set_ylim(0.5, max(base.max(), real.max(), 1.0) * 8)
    else:
        axes.set_xlim(edges[0], edges[-1])
        axes.set_ylim(0, max(base.max(), real.max(), 1.0) * top_factor)
        axes.yaxis.set_major_locator(MaxNLocator(4, prune="lower"))
        axes.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    if variable == "phi":
        ratio_axes.set_xticks(np.pi * np.array([-1, -0.5, 0, 0.5, 1]))
        ratio_axes.set_xticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])
    axes.tick_params(axis="x", which="both", labelbottom=False)
    low, high = ratio_axis_range(ratios) if ratio_range is None else ratio_range
    ratio_axes.set_ylim(low, high)
    set_ratio_ticks(ratio_axes, low, high)
    return low, high


def kinematics_legend(figure: Figure, colors: dict[float, str], baseline_label: str, top: float) -> None:
    """One row above the panels: real data, the baseline, and each alpha; `top` in figure coordinates."""
    handles = [Line2D([], [], color=REAL_DATA_COLOR, alpha=REAL_DATA_OPACITY, linewidth=HISTOGRAM_LINE_WIDTH,
                      label="real data"),
               Line2D([], [], color=BASELINE_COLOR, linewidth=HISTOGRAM_LINE_WIDTH, linestyle="--",
                      label=baseline_label)]
    handles += [Line2D([], [], color=color, linewidth=HISTOGRAM_LINE_WIDTH, label=alpha_label(alpha))
                for alpha, color in colors.items()]
    figure.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.53, top), ncol=len(handles),
                  columnspacing=KINEMATICS_LEGEND_SPACING, handlelength=2.0, borderaxespad=0.1)


def checked_rows(rows: dict[int, dict[str, Panel]], variables: tuple[str, ...],
                 optional: tuple[str, ...] = ()) -> list[float]:
    """The alphas of `rows`, the same in every panel; stops on a missing variable of `variables` or a bad panel. A
    variable of `optional` is checked in the rows that have it."""
    if not rows:
        raise ValueError("expected at least one particle type, got none")
    alphas = None
    for code, panels in rows.items():
        if code not in PARTICLE_LABELS:
            raise ValueError(f"particle type {code}: not one of ShellFlow's codes {sorted(PARTICLE_LABELS)}")
        missing = [variable for variable in variables if variable not in panels]
        if missing:
            raise ValueError(f"particle type {code}: no panel of {missing}")
        for variable in (*variables, *(variable for variable in optional if variable in panels)):
            check_panel(panels[variable], f"particle type {code}, {variable}")
            if alphas is None:
                alphas = sorted(panels[variable].steered)
            elif sorted(panels[variable].steered) != alphas:
                raise ValueError(f"particle type {code}, {variable}: alphas {sorted(panels[variable].steered)}, "
                                 f"other panels have {alphas}")
    return alphas


def kinematics_figure(panels: dict[str, Panel], *, particle: int, baseline_label: str,
                      ratio_min_baseline: int) -> Figure:
    """Fig. 4 of the paper: p_T, eta, and phi of the particles of type `particle` (13, the edited muons), real data,
    baseline, and each alpha in counts, with steered over baseline below each panel.

    panels: variable -> Panel from `kinematics.type_panels`, with at least "pt", "eta", and "phi". ratio_min_baseline:
    the ratio panels leave out bins with fewer baseline particles; the paper draws every bin with one (1).
    baseline_label: legend entry of the baseline, r"$\\mathtt{ShellFlow}$" in the paper.
    """
    variables = ("pt", "eta", "phi")
    alphas = checked_rows({particle: panels}, variables)
    colors = alpha_colors(alphas)
    with style.context(STYLE_FILE):
        figure = Figure(figsize=(TEXT_WIDTH, KINEMATICS_HEIGHT))
        grid = figure.add_gridspec(2, len(variables), height_ratios=[2.3, 1.0], hspace=0.0, wspace=0.30, left=0.075,
                                   right=0.995, bottom=0.15, top=0.81)
        for column, variable in enumerate(variables):
            axes = figure.add_subplot(grid[0, column])
            ratio_axes = figure.add_subplot(grid[1, column], sharex=axes)
            draw_kinematics_panel(axes, ratio_axes, variable, panels[variable], colors,
                                  min_baseline=ratio_min_baseline, x_margin=1.3, top_factor=1.45,
                                  start_below_decade=False, ratio_range=None)
            right = variable in LOG_VARIABLES
            axes.text(0.96 if right else 0.5, 0.95, PARTICLE_LABELS[particle][0], transform=axes.transAxes,
                      va="top", ha="right" if right else "center", fontsize=7)
            ratio_axes.set_xlabel(KINEMATICS_LABELS[variable], labelpad=1)
            if column == 0:
                axes.set_ylabel("counts", labelpad=1)
                ratio_axes.set_ylabel("ratio", labelpad=1)
        kinematics_legend(figure, colors, baseline_label, 0.885)
        fit_to_width(figure, TEXT_WIDTH)
    return figure


def all_particles_figure(rows: dict[int, dict[str, Panel]], *, baseline_label: str, ratio_min_baseline: int,
                         ratio_range: tuple[float, float]) -> Figure:
    """Fig. 14 of the paper (App. D.1): p_T, eta, phi, E, and m of each particle type in `rows`, one row per type with
    a bold title, real data, baseline, and each alpha in counts, with steered over baseline below each panel.

    rows: ShellFlow type code -> variable -> Panel from `kinematics.type_panels`, plus "m" from
    `kinematics.mass_panels` for the massive types, in the order of the rows; the paper shows electrons, photons, jets,
    large-R jets, and hadronic taus. If a row has an m panel, every row has five aligned columns and the m slot of the
    other rows stays empty, as in the paper; without any m panel the figure has four columns. Each x title sits under
    the last row with that column. ratio_range: (low, high) of every ratio panel; bins outside run off the panel;
    (0.9, 1.1) in the paper. ratio_min_baseline and baseline_label: as in `kinematics_figure`.
    """
    alphas = checked_rows(rows, VARIABLES, optional=(MASS,))
    columns = (*VARIABLES, MASS) if any(MASS in panels for panels in rows.values()) else VARIABLES
    last_row = {variable: max(index for index, panels in enumerate(rows.values()) if variable in panels)
                for variable in columns}
    if len(ratio_range) != 2 or not 0 <= ratio_range[0] < 1 < ratio_range[1]:
        raise ValueError(f"ratio_range: expected (low, high) with 0 <= low < 1 < high, got {ratio_range}")
    colors = alpha_colors(alphas)
    row_height = ROW_TITLE + ROW_GAP + ROW_PANELS + ROW_TICKS
    height = ROW_LEGEND + row_height * len(rows) + ROW_FOOT
    with style.context(STYLE_FILE):
        figure = Figure(figsize=(TEXT_WIDTH, height))
        down = lambda inches: 1 - inches / height       # inches from the top -> figure coordinates
        for index, (code, panels) in enumerate(rows.items()):
            row_top = ROW_LEGEND + index * row_height
            figure.text(0.53, down(row_top), PARTICLE_LABELS[code][1], ha="center", va="top", fontsize=ROW_TITLE_SIZE)
            panels_top = row_top + ROW_TITLE + ROW_GAP
            grid = figure.add_gridspec(2, len(columns), height_ratios=[2.0, 1.0], hspace=0.0, wspace=0.38,
                                       left=0.07, right=0.995, top=down(panels_top),
                                       bottom=down(panels_top + ROW_PANELS))
            for column, variable in enumerate(columns):
                if variable not in panels:
                    continue
                axes = figure.add_subplot(grid[0, column])
                ratio_axes = figure.add_subplot(grid[1, column], sharex=axes)
                draw_kinematics_panel(axes, ratio_axes, variable, panels[variable], colors,
                                      min_baseline=ratio_min_baseline, x_margin=1.8, top_factor=1.3,
                                      start_below_decade=True, ratio_range=ratio_range)
                if index == last_row[variable]:
                    ratio_axes.set_xlabel(KINEMATICS_LABELS[variable], labelpad=1)
                if column == 0:
                    axes.set_ylabel("counts", labelpad=1)
                    ratio_axes.set_ylabel("ratio", labelpad=1)
        kinematics_legend(figure, colors, baseline_label, down(ROW_LEGEND))
        fit_to_width(figure, TEXT_WIDTH)
    return figure
