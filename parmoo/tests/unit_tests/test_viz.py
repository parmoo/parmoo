""" Unit tests for the viz layer.

The viz tests stay in one file, but the repeated call patterns are factored
into helpers and parametrized tables so the intent is easier to scan.
"""

from parmoo.viz.plot import parallel_coordinates, radar, scatter

import pytest


# Mark these tests as requiring the additional chrome dependency.
# Get it using the ``plotly_get_chrome`` command.
pytestmark = pytest.mark.requires_chrome

PLOTTERS = [scatter, parallel_coordinates, radar]
PLOT_NAMES = {
    scatter: "scatter",
    parallel_coordinates: "parallel_coordinates",
    radar: "radar",
}
PLOT_TYPES = {
    scatter: "scatter",
    parallel_coordinates: "parallel",
    radar: "radar",
}
EXPECTED_FILENAMES = {
    "pf": "Pareto Front.html",
    "obj": "Objective Data.html",
}

STATIC_EXPORT_CASES = [
    pytest.param(scatter, "html", "Pareto Front.html", id="scatter-html"),
    pytest.param(parallel_coordinates, "svg", "Pareto Front.svg",
                 id="parallel-svg"),
    pytest.param(radar, "pdf", "Pareto Front.pdf", id="radar-pdf"),
    pytest.param(scatter, "jpeg", "Pareto Front.jpeg", id="scatter-jpeg"),
    pytest.param(parallel_coordinates, "png", "Pareto Front.png",
                 id="parallel-png"),
    pytest.param(radar, "webp", "Pareto Front.webp", id="radar-webp"),
]

DATABASE_CASES = [
    pytest.param(
        plotter,
        moop_name,
        db,
        points,
        id=f"{PLOT_NAMES[plotter]}-{moop_name}-{db}-{points}",
    )
    for plotter in PLOTTERS
    for moop_name in ("quickstart_moop", "dtlz2_moop")
    for db in ("pf", "obj")
    for points in (
        "constraint_satisfying",
        "constraint_violating",
        "all",
        "none",
    )
]

INVALID_DB_CASES = [
    pytest.param("asdf", id="db-string"),
    pytest.param(1234, id="db-int"),
]

INVALID_OUTPUT_CASES = [
    pytest.param("asdf", id="output-string"),
    pytest.param(1234, id="output-int"),
]

INVALID_POINTS_CASES = [
    pytest.param("asdf", id="points-string"),
    pytest.param(1234, id="points-int"),
]

INVALID_NUMERIC_CASES = [
    pytest.param("height", "asdf", id="height-string"),
    pytest.param("width", "asdf", id="width-string"),
    pytest.param("fontsize", -1, id="fontsize-low"),
    pytest.param("fontsize", 101, id="fontsize-high"),
]

NUMERIC_CASES = [
    pytest.param("height", 1, id="height-1"),
    pytest.param("height", 50, id="height-50"),
    pytest.param("height", 92.2678, id="height-float"),
    pytest.param("height", 7789, id="height-large"),
    pytest.param("width", 1, id="width-1"),
    pytest.param("width", 50, id="width-50"),
    pytest.param("width", 92.2678, id="width-float"),
    pytest.param("width", 7789, id="width-large"),
    pytest.param("fontsize", 1, id="fontsize-1"),
    pytest.param("fontsize", 50, id="fontsize-50"),
    pytest.param("fontsize", 55.71, id="fontsize-float"),
    pytest.param("fontsize", 100, id="fontsize-100"),
]

STYLE_CASES = [
    pytest.param("font", "Verdana", id="font-verdana"),
    pytest.param("font", "Times New Roman", id="font-times"),
    pytest.param("background_color", "white", id="background-white"),
    pytest.param("background_color", "black", id="background-black"),
    pytest.param("background_color", "transparent", id="background-clear"),
    pytest.param("background_color", "grey", id="background-grey"),
    pytest.param("screenshot", "png", id="screenshot-png"),
    pytest.param("screenshot", "jpeg", id="screenshot-jpeg"),
    pytest.param("screenshot", "svg", id="screenshot-svg"),
    pytest.param("screenshot", "webp", id="screenshot-webp"),
    pytest.param("image_export_format", "html", id="image-html"),
    pytest.param("image_export_format", "svg", id="image-svg"),
    pytest.param("image_export_format", "pdf", id="image-pdf"),
    pytest.param("image_export_format", "png", id="image-png"),
    pytest.param("image_export_format", "jpeg", id="image-jpeg"),
    pytest.param("image_export_format", "webp", id="image-webp"),
    pytest.param("data_export_format", "csv", id="data-csv"),
    pytest.param("data_export_format", "json", id="data-json"),
    pytest.param("dev_mode", True, id="dev-mode-on"),
    pytest.param("dev_mode", False, id="dev-mode-off"),
    pytest.param("pop_up", True, id="pop-up-on"),
    pytest.param("pop_up", False, id="pop-up-off"),
    pytest.param("port", "http://127.0.0.1:8050/", id="port-default"),
]

INVALID_EXPORT_CASES = [
    pytest.param("screenshot", "asdf", id="screenshot-string"),
    pytest.param("screenshot", 1234, id="screenshot-int"),
    pytest.param("image_export_format", "asdf", id="image-string"),
    pytest.param("image_export_format", 1234, id="image-int"),
    pytest.param("data_export_format", "asdf", id="data-string"),
    pytest.param("data_export_format", 1234, id="data-int"),
    pytest.param("port", "asdf", id="port-string"),
    pytest.param("port", 1234, id="port-int"),
]


def _assert_output(plotter, moop, expected_name, monkeypatch, tmp_path, **kwargs):
    """Call a plotter and assert it writes the expected output file."""

    monkeypatch.chdir(tmp_path)
    plotter(moop, **kwargs)
    assert (tmp_path / expected_name).is_file()


@pytest.mark.parametrize("plotter, output, expected_name", STATIC_EXPORT_CASES)
def test_static_export(plotter, output, expected_name, quickstart_moop,
                       monkeypatch, tmp_path):
    """Create a MOOP object and export various plot types."""

    _assert_output(
        plotter,
        quickstart_moop,
        expected_name,
        monkeypatch,
        tmp_path,
        output=output,
    )


@pytest.mark.parametrize("plotter, moop_name, db, points", DATABASE_CASES)
def test_database_options(request, plotter, moop_name, db, points, monkeypatch,
                          tmp_path):
    """Plot both databases with the supported constraint filters."""

    moop = request.getfixturevalue(moop_name)
    _assert_output(
        plotter,
        moop,
        EXPECTED_FILENAMES[db],
        monkeypatch,
        tmp_path,
        db=db,
        points=points,
        output="html",
    )


@pytest.mark.parametrize("plotter", PLOTTERS, ids=lambda plotter: PLOT_NAMES[plotter])
def test_dash_dispatch(plotter, quickstart_moop, monkeypatch):
    """Check that the wrapper dispatches to the Dash app for interactive use."""

    captured = {}

    def fake_dash_app(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs

    monkeypatch.setattr("parmoo.viz.plot.Dash_App", fake_dash_app)
    plotter(quickstart_moop, output="dash")

    kwargs = captured["kwargs"]
    assert kwargs["plot_type"] == PLOT_TYPES[plotter]
    assert kwargs["moop"] is quickstart_moop
    assert kwargs["db"] == "pf"
    assert kwargs["points"] == "constraint_satisfying"
    assert kwargs["height"] == "auto"
    assert kwargs["width"] == "auto"
    assert kwargs["font"] == "auto"
    assert kwargs["fontsize"] == "auto"
    assert kwargs["background_color"] == "auto"
    assert kwargs["screenshot"] == "svg"
    assert kwargs["image_export_format"] == "svg"
    assert kwargs["data_export_format"] == "csv"
    assert kwargs["dev_mode"] is False
    assert kwargs["pop_up"] is True
    assert kwargs["port"] == "http://127.0.0.1:8050/"


@pytest.mark.parametrize("bad_db", INVALID_DB_CASES)
def test_invalid_db_values(quickstart_moop, bad_db):
    with pytest.raises(ValueError):
        scatter(quickstart_moop, db=bad_db)


@pytest.mark.parametrize("bad_output", INVALID_OUTPUT_CASES)
def test_invalid_output_values(quickstart_moop, bad_output):
    with pytest.raises(ValueError):
        scatter(quickstart_moop, output=bad_output)


@pytest.mark.parametrize("bad_points", INVALID_POINTS_CASES)
def test_invalid_points_values(quickstart_moop, bad_points):
    with pytest.raises(ValueError):
        scatter(quickstart_moop, output="html", points=bad_points)


@pytest.mark.parametrize("keyword, value", INVALID_NUMERIC_CASES)
def test_invalid_numeric_values(quickstart_moop, keyword, value):
    with pytest.raises(ValueError):
        scatter(quickstart_moop, output="html", **{keyword: value})


@pytest.mark.parametrize("keyword, value", NUMERIC_CASES)
def test_numeric_values_render(quickstart_moop, keyword, value, monkeypatch,
                               tmp_path):
    _assert_output(
        scatter,
        quickstart_moop,
        "Pareto Front.html",
        monkeypatch,
        tmp_path,
        output="html",
        **{keyword: value},
    )


@pytest.mark.parametrize("keyword, value", STYLE_CASES)
def test_style_values_render(quickstart_moop, keyword, value, monkeypatch,
                             tmp_path):
    _assert_output(
        scatter,
        quickstart_moop,
        "Pareto Front.html",
        monkeypatch,
        tmp_path,
        output="html",
        **{keyword: value},
    )


@pytest.mark.parametrize("keyword, value", INVALID_EXPORT_CASES)
def test_invalid_export_values(quickstart_moop, keyword, value):
    with pytest.raises(ValueError):
        scatter(quickstart_moop, output="html", **{keyword: value})


@pytest.fixture(scope="module")
def quickstart_moop():
    """ A solved 2-objective MOOP with a constraint, from the quickstart.

    Module scoped because solve() is expensive and the plotting functions
    only read from the MOOP.

    """

    import numpy as np
    from parmoo import MOOP
    from parmoo.searches import LatinHypercube
    from parmoo.surrogates import GaussRBF
    from parmoo.acquisitions import UniformWeights
    from parmoo.optimizers import GlobalSurrogate_PS

    my_moop = MOOP(GlobalSurrogate_PS)

    my_moop.addDesign({
        'name': "x1",
        'des_type': "continuous",
        'lb': 0.0, 'ub': 1.0
    })
    my_moop.addDesign({
        'name': "x2",
        'des_type': "categorical",
        'levels': 3
    })

    def sim_func(x):
        if x["x2"] == 0:
            return np.array([(x["x1"] - 0.2) ** 2, (x["x1"] - 0.8) ** 2])
        else:
            return np.array([99.9, 99.9])

    my_moop.addSimulation({
        'name': "MySim",
        'm': 2,
        'sim_func': sim_func,
        'search': LatinHypercube,
        'surrogate': GaussRBF,
        'hyperparams': {'search_budget': 10}
    })

    my_moop.addObjective({
        'name': "Cost of driving",
        'obj_func': lambda x, s: s["MySim"][0]
    })
    my_moop.addObjective({
        'name': "Time spent driving",
        'obj_func': lambda x, s: s["MySim"][1]
    })

    my_moop.addConstraint({
        'name': "c1",
        'constraint': lambda x, s: 0.1 - x["x1"]
    })

    for i in range(2):
        my_moop.addAcquisition({
            'acquisition': UniformWeights,
            'hyperparams': {}
        })

    my_moop.solve(1)

    return my_moop


@pytest.fixture(scope="module")
def dtlz2_moop():
    """ A solved 5-objective DTLZ2 MOOP with no constraints.

    Module scoped for the same reason as quickstart_moop.

    """

    from parmoo import MOOP
    from parmoo.acquisitions import RandomConstraint
    from parmoo.searches import LatinHypercube
    from parmoo.surrogates import GaussRBF
    from parmoo.optimizers import GlobalSurrogate_BFGS
    from parmoo.objectives.dtlz import dtlz2_obj, dtlz2_grad
    from parmoo.simulations.dtlz import g2_sim

    n = 6  # number of design variables
    o = 5  # number of objectives
    q = 2  # batch size (number of acquisitions)
    # Create MOOP
    moop = MOOP(GlobalSurrogate_BFGS)
    # Add n design variables
    for i in range(n):
        moop.addDesign({
            'name': f"Input {i+1}",
            'des_type': 'continuous',
            'lb': 0.0,
            'ub': 1.0,
            'des_tol': 1.0e-8
        })
    # Create the g2 simulation
    moop.addSimulation({
        'name': "g2",
        'm': 1,
        'sim_func': g2_sim(
            moop.getDesignType(),
            num_obj=o,
            offset=0.5
        ),
        'search': LatinHypercube,
        'surrogate': GaussRBF,
        'hyperparams': {'search_budget': 10}
    })
    # Add o objectives
    for i in range(o):
        moop.addObjective({
            'name': f"Objective {i+1}",
            'obj_func': dtlz2_obj(
                moop.getDesignType(),
                moop.getSimulationType(),
                i, num_obj=o),
            'obj_grad': dtlz2_grad(
                moop.getDesignType(),
                moop.getSimulationType(),
                i, num_obj=o)
        })
    # Add q acquisition functions
    for i in range(q):
        moop.addAcquisition({'acquisition': RandomConstraint})
    # Solve the MOOP with 20 iterations
    moop.solve(1)
    return moop


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
