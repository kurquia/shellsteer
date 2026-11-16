import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from matplotlib import colormaps, style
from matplotlib.colors import to_hex
from matplotlib.container import ErrorbarContainer
from matplotlib.patches import StepPatch

from shellsteer.plots import (BASELINE_COLOR, COLUMN_WIDTH, PAD, POLE_AND_WIDTH_WIDTH, REAL_DATA_COLOR, STYLE_FILE,
                              TEXT_WIDTH, pole_and_width_figure, save, selectivity_figure, steered_spectrum_figure)

ALPHAS = (-2.0, -1.0, 1.0, 2.0)
ZOOMS = {r"$J/\psi$": (2.0, 4.5), r"$\Upsilon$": (6.0, 14.0), "$Z$": (50.0, 140.0)}
SPECTRUM_SETTINGS = dict(edges=np.geomspace(0.3, 300, 41), zooms=ZOOMS, zoom_bins=20,
                         baseline_label=r"$\mathtt{ShellFlow}$", pair_label=r"\mu\mu")


def masses(seed: int, n: int = 3000) -> np.ndarray:
    """A continuum from 0.3 to 300 GeV plus peaks at 3.1, 9.5, and 91 GeV."""
    rng = np.random.default_rng(seed)
    continuum = np.exp(rng.uniform(np.log(0.3), np.log(300), n // 2))
    peaks = rng.normal(np.repeat([3.1, 9.5, 91.0], n // 6 + 1)[: n - n // 2], [0.05])
    return rng.permutation(np.concatenate([continuum, peaks]))


def spectrum_inputs(real: bool = True):
    baseline = masses(0)
    steered = {alpha: baseline * np.exp(0.12 * alpha * (np.abs(baseline - 91) < 20)) for alpha in ALPHAS}
    return baseline, steered, masses(1) if real else None


def printed_width(figure) -> float:
    with style.context(STYLE_FILE):         # as `save` draws it
        return figure.get_tightbbox().width + 2 * PAD


def steps(axes) -> list[StepPatch]:
    return [patch for patch in axes.patches if isinstance(patch, StepPatch)]


def winter(alphas) -> list[str]:
    return [to_hex(colormaps["winter"](x)) for x in np.linspace(0, 1, len(alphas))]


def test_spectrum_panels_and_lines():
    figure = steered_spectrum_figure(*spectrum_inputs(), **SPECTRUM_SETTINGS)
    spectrum, ratio, *zooms = figure.axes
    assert len(zooms) == len(ZOOMS)
    assert [zoom.texts[0].get_text() for zoom in zooms] == list(ZOOMS)
    assert [zoom.get_xlim() for zoom in zooms] == list(ZOOMS.values())
    for axes in (spectrum, *zooms):
        lines = steps(axes)
        assert len(lines) == 2 + len(ALPHAS)                       # real data, baseline, one per alpha
        real, baseline, *steered = lines
        assert to_hex(real.get_edgecolor()) == to_hex(REAL_DATA_COLOR) and real.get_alpha() == 0.6
        assert to_hex(baseline.get_edgecolor()) == to_hex(BASELINE_COLOR) and baseline.get_linestyle() == "--"
        assert baseline.get_zorder() > max(line.get_zorder() for line in (real, *steered))
        assert [to_hex(line.get_edgecolor()) for line in steered] == winter(ALPHAS)
        assert all(line.get_linestyle() in ("-", "solid") for line in steered)
    assert len(steps(ratio)) == len(ALPHAS) and ratio.get_yscale() == "log"
    assert [text.get_text() for text in spectrum.get_legend().get_texts()] == [
        "real data", r"$\mathtt{ShellFlow}$", r"$\alpha = -2$", r"$\alpha = -1$", r"$\alpha = +1$", r"$\alpha = +2$"]
    assert spectrum.get_xlim() == pytest.approx((0.3, 300))
    assert printed_width(figure) == pytest.approx(TEXT_WIDTH, abs=0.005)


def test_spectrum_without_real_data():
    figure = steered_spectrum_figure(*spectrum_inputs(real=False), **SPECTRUM_SETTINGS)
    spectrum, _, *zooms = figure.axes
    for axes in (spectrum, *zooms):
        assert len(steps(axes)) == 1 + len(ALPHAS)
        assert to_hex(REAL_DATA_COLOR) not in [to_hex(line.get_edgecolor()) for line in steps(axes)]
    assert "real data" not in [text.get_text() for text in spectrum.get_legend().get_texts()]


def test_spectrum_leaves_global_settings_alone():
    before = dict(matplotlib.rcParams)
    figure = steered_spectrum_figure(*spectrum_inputs(), **SPECTRUM_SETTINGS)
    assert dict(matplotlib.rcParams) == before
    assert figure.axes[0].yaxis.label.get_fontsize() == 8          # paper.mplstyle applied inside the function


def test_spectrum_stops_on_bad_inputs():
    baseline, steered, real = spectrum_inputs()
    cases = [
        ((baseline, {**steered, 1.0: baseline[:-1]}, real), {}, r"steered\[\+1\]: 2999 masses, the baseline has 3000"),
        ((baseline, steered, real[:-1]), {}, "real_data: 2999 masses, the baseline has 3000"),
        ((baseline, {0.0: baseline}, real), {}, "alpha 0 is the baseline"),
        ((baseline, {}, real), {}, "expected at least one alpha"),
        ((baseline[:, None], steered, real), {}, r"baseline: expected a 1-D array of masses, got shape \(3000, 1\)"),
        ((baseline, steered, real), {"edges": np.geomspace(300, 0.3, 41)}, "expected increasing edges"),
        ((baseline, steered, real), {"edges": np.linspace(0, 300, 41)}, "expected finite positive edges"),
        ((baseline, steered, real), {"zooms": {"$Z$": (140, 50)}}, r"zooms\['\$Z\$'\]: expected \(low, high\)"),
        ((baseline, steered, real), {"zooms": {}}, "expected at least one zoom"),
        ((baseline, steered, real), {"zoom_bins": 0}, "zoom_bins: expected a positive integer, got 0"),
    ]
    for arguments, changes, message in cases:
        with pytest.raises(ValueError, match=message):
            steered_spectrum_figure(*arguments, **{**SPECTRUM_SETTINGS, **changes})


def curve(alphas, scale):
    alphas = np.asarray(alphas, dtype=float)
    return {"alpha": alphas, "target shift": 0.12 * alphas, "target shift std": np.full(len(alphas), 1e-3),
            "R_60": scale * np.abs(alphas), "R_60 std": np.full(len(alphas), 2e-3)}


SELECTIVITY_SETTINGS = dict(target_label="$Z$", pair_label=r"\mu\mu", max_mass=60)


def test_selectivity_points_and_lines():
    curves = {"FKT": curve(ALPHAS, 0.03), "Fisher": curve((-4, -3, 3, 4), 0.12), "positive only": curve((1, 2), 0.2)}
    figure = selectivity_figure(curves, **SELECTIVITY_SETTINGS)
    (axes,) = figure.axes
    containers = [c for c in axes.containers if isinstance(c, ErrorbarContainer)]
    assert len(containers) == 5                    # FKT and Fisher: negative and positive alpha apart
    for container in containers:
        shift = container.lines[0].get_xdata()
        assert np.all(shift < 0) or np.all(shift > 0)                  # no line crosses shift 0
    points = {}
    for container in containers:
        color = to_hex(container.lines[0].get_color())
        points[color] = points.get(color, 0) + len(container.lines[0].get_xdata())
    assert list(points.values()) == [4, 4, 2]                         # one marker per alpha, one colour per curve
    assert [text.get_text() for text in axes.get_legend().get_texts()] == list(curves)
    assert "60 GeV" in axes.get_ylabel() and r"\Delta\log m_{\mu\mu}" in axes.get_xlabel()
    assert printed_width(figure) == pytest.approx(COLUMN_WIDTH, abs=0.005)


def test_selectivity_stops_on_bad_inputs():
    good = curve(ALPHAS, 0.03)
    missing = {key: value for key, value in good.items() if key != "R_60 std"}
    cases = [
        ({"FKT": missing}, "expected exactly"),
        ({"FKT": {**good, "R_60 mean": good["R_60"]}}, "expected exactly"),
        ({"FKT": {**good, "R_60": good["R_60"][:3]}}, "expected 1-D arrays of one length"),
        ({"FKT": curve((-1, 0, 1), 0.03)}, "alpha 0 is the baseline"),
        ({"FKT": curve((1, 1, 2), 0.03)}, r"alpha \[1.\] appears more than once"),
        ({"FKT": {**good, "R_60 std": -good["R_60 std"]}}, "'R_60 std' has negative standard deviations"),
        ({}, "expected at least one curve"),
        ({f"curve {i}": good for i in range(6)}, "6 curves, the figure has colours for 5"),
    ]
    for curves, message in cases:
        with pytest.raises(ValueError, match=message):
            selectivity_figure(curves, **SELECTIVITY_SETTINGS)


def pole_and_width_inputs():
    alpha = np.arange(-3, 3.5, 0.5)
    rng = np.random.default_rng(2)
    return (alpha, 90.6 * np.exp(0.122 * alpha) + rng.normal(0, 0.3, len(alpha)), np.full(len(alpha), 0.05),
            2.4 + 0.4 * alpha ** 2, np.full(len(alpha), 0.05))


@pytest.mark.parametrize("slope", [0.122, None])
def test_pole_and_width_panels(slope):
    alpha, pole, pole_std, width, width_std = pole_and_width_inputs()
    figure = pole_and_width_figure(alpha, pole, pole_std, width, width_std, first_order_log_slope=slope)
    pole_panel, width_panel = figure.axes
    for axes, values in ((pole_panel, pole), (width_panel, width)):
        (container,) = axes.containers
        assert np.array_equal(container.lines[0].get_xdata(), alpha)    # one point per alpha, the baseline at 0
        assert np.array_equal(container.lines[0].get_ydata(), values)
        dotted = [line for line in axes.get_lines() if line.get_linestyle() == ":"]
        assert len(dotted) == (0 if slope is None else 1)
        if slope is not None:
            x, y = dotted[0].get_data()
            assert np.allclose(y, values[alpha == 0] * np.exp(slope * x))
    assert [axes.texts[0].get_text() for axes in figure.axes] == ["(a)", "(b)"]
    assert printed_width(figure) == pytest.approx(POLE_AND_WIDTH_WIDTH, abs=0.005)


def test_pole_and_width_stops_on_bad_inputs():
    alpha, pole, pole_std, width, width_std = pole_and_width_inputs()
    cases = [
        ((alpha + 0.25, pole, pole_std, width, width_std), "expected one alpha 0 for the baseline fit"),
        ((alpha, pole[:-1], pole_std, width, width_std), "expected 1-D arrays of one length"),
        ((np.where(alpha == 3, 2.5, alpha), pole, pole_std, width, width_std), r"alpha \[2.5\] appears more"),
        ((alpha, pole, -pole_std, width, width_std), "'pole std' has negative standard deviations"),
        ((alpha, pole, pole_std, np.where(alpha == 1, np.nan, width), width_std), "'width' has non-finite values"),
    ]
    for arguments, message in cases:
        with pytest.raises(ValueError, match=message):
            pole_and_width_figure(*arguments, first_order_log_slope=0.122)


def test_save_writes_a_pdf(tmp_path):
    figure = pole_and_width_figure(*pole_and_width_inputs(), first_order_log_slope=None)
    save(figure, tmp_path / "pole_and_width.pdf")
    assert (tmp_path / "pole_and_width.pdf").read_bytes().startswith(b"%PDF")
    with pytest.raises(ValueError, match="does not end in .pdf"):
        save(figure, tmp_path / "pole_and_width.png")
