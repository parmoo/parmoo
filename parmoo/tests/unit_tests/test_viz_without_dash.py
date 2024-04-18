def test_static_export():
    """ Create a MOOP object and export various plot types.

    Tests all export formats except eps
    (because eps may not be supported on all machines).

    """

    from parmoo.viz.plot import (
        scatter,
        parallel_coordinates,
        radar,
    )
    import os

    # Pre-calculate a moop object
    moop1 = run_quickstart()

    # * html output
    scatter(moop1, output='html')
    assert (os.path.exists("Pareto Front.html"))
    os.remove("Pareto Front.html")

    # * svg output
    parallel_coordinates(moop1, output='svg')
    assert (os.path.exists("Pareto Front.svg"))
    os.remove("Pareto Front.svg")

    # * pdf output
    radar(moop1, output='pdf')
    assert (os.path.exists("Pareto Front.pdf"))
    os.remove("Pareto Front.pdf")

    # * jpeg output
    scatter(moop1, output='jpeg')
    assert (os.path.exists("Pareto Front.jpeg"))
    os.remove("Pareto Front.jpeg")

    # * png output
    parallel_coordinates(moop1, output='png')
    assert (os.path.exists("Pareto Front.png"))
    os.remove("Pareto Front.png")

    # * webp output
    radar(moop1, output='webp')
    assert (os.path.exists("Pareto Front.webp"))
    os.remove("Pareto Front.webp")


def test_quantity_constraints_objectives():
    """ Create a MOOP object and plot PF with and w/o constraint data. """

    from parmoo.viz.plot import (
        scatter,
        parallel_coordinates,
        radar,
    )
    import os

    # Pre-calculate two moop objects
    moop1 = run_quickstart()
    moop2 = run_dtlz2()

    # * 2 objective scatter with constraint
    scatter(moop1, output='html')
    assert (os.path.exists("Pareto Front.html"))
    os.remove("Pareto Front.html")

    # * 2 objective parallel with constraint
    parallel_coordinates(moop1, output='html')
    assert (os.path.exists("Pareto Front.html"))
    os.remove("Pareto Front.html")

    # * 2 objective radar with constraint
    radar(moop1, output='html')
    assert (os.path.exists("Pareto Front.html"))
    os.remove("Pareto Front.html")

    # * 5 objective scatter without constraint
    scatter(moop2, output='html')
    assert (os.path.exists("Pareto Front.html"))
    os.remove("Pareto Front.html")

    # * 5 objective parallel without constraint
    parallel_coordinates(moop2, output='html')
    assert (os.path.exists("Pareto Front.html"))
    os.remove("Pareto Front.html")

    # * 5 objective radar without constraint
    radar(moop2, output='html')
    assert (os.path.exists("Pareto Front.html"))
    os.remove("Pareto Front.html")


def test_database_options():
    """ Create a MOOP object and plot the full database.

    Generates plots with and without constraint violations.

    """

    from parmoo.viz.plot import (
        scatter,
        parallel_coordinates,
        radar,
    )
    import os

    # Pre-calculate two moop objects
    moop1 = run_quickstart()
    moop2 = run_dtlz2()

    # * pf x constraint_satisfying x constraints in MOOP
    scatter(
        moop1,
        db='pf',
        points='constraint_satisfying',
        output='html')
    assert (os.path.exists("Pareto Front.html"))
    os.remove("Pareto Front.html")

    parallel_coordinates(
        moop1,
        db='pf',
        points='constraint_satisfying',
        output='html')
    assert (os.path.exists("Pareto Front.html"))
    os.remove("Pareto Front.html")

    radar(
        moop1,
        db='pf',
        points='constraint_satisfying',
        output='html')
    assert (os.path.exists("Pareto Front.html"))
    os.remove("Pareto Front.html")

    # * obj x constraint_satisfying x constraints in MOOP
    scatter(
        moop1,
        db='obj',
        points='constraint_satisfying',
        output='html')
    assert (os.path.exists("Objective Data.html"))
    os.remove("Objective Data.html")

    parallel_coordinates(
        moop1,
        db='obj',
        points='constraint_satisfying',
        output='html')
    assert (os.path.exists("Objective Data.html"))
    os.remove("Objective Data.html")

    radar(
        moop1,
        db='obj',
        points='constraint_satisfying',
        output='html')
    assert (os.path.exists("Objective Data.html"))
    os.remove("Objective Data.html")

    # * pf x constraint_violating x constraints in MOOP
    scatter(
        moop1,
        db='pf',
        points='constraint_violating',
        output='html')
    assert (os.path.exists("Pareto Front.html"))
    os.remove("Pareto Front.html")

    parallel_coordinates(
        moop1,
        db='pf',
        points='constraint_violating',
        output='html')
    assert (os.path.exists("Pareto Front.html"))
    os.remove("Pareto Front.html")

    radar(
        moop1,
        db='pf',
        points='constraint_violating',
        output='html')
    assert (os.path.exists("Pareto Front.html"))
    os.remove("Pareto Front.html")

    # * obj x constraint_violating x constraints in MOOP
    scatter(
        moop1,
        db='obj',
        points='constraint_violating',
        output='html')
    assert (os.path.exists("Objective Data.html"))
    os.remove("Objective Data.html")

    parallel_coordinates(
        moop1,
        db='obj',
        points='constraint_violating',
        output='html')
    assert (os.path.exists("Objective Data.html"))
    os.remove("Objective Data.html")

    radar(
        moop1,
        db='obj',
        points='constraint_violating',
        output='html')
    assert (os.path.exists("Objective Data.html"))
    os.remove("Objective Data.html")

    # * pf x all x constraints in MOOP
    scatter(
        moop1,
        db='pf',
        points='all',
        output='html')
    assert (os.path.exists("Pareto Front.html"))
    os.remove("Pareto Front.html")

    parallel_coordinates(
        moop1,
        db='pf',
        points='all',
        output='html')
    assert (os.path.exists("Pareto Front.html"))
    os.remove("Pareto Front.html")

    radar(
        moop1,
        db='pf',
        points='all',
        output='html')
    assert (os.path.exists("Pareto Front.html"))
    os.remove("Pareto Front.html")

    # * obj x all x constraints in MOOP
    scatter(
        moop1,
        db='obj',
        points='all',
        output='html')
    assert (os.path.exists("Objective Data.html"))
    os.remove("Objective Data.html")

    parallel_coordinates(
        moop1,
        db='obj',
        points='all',
        output='html')
    assert (os.path.exists("Objective Data.html"))
    os.remove("Objective Data.html")

    radar(
        moop1,
        db='obj',
        points='all',
        output='html')
    assert (os.path.exists("Objective Data.html"))
    os.remove("Objective Data.html")

    # * pf x none x constraints in MOOP
    scatter(
        moop1,
        db='pf',
        points='none',
        output='html')
    assert (os.path.exists("Pareto Front.html"))
    os.remove("Pareto Front.html")

    parallel_coordinates(
        moop1,
        db='pf',
        points='none',
        output='html')
    assert (os.path.exists("Pareto Front.html"))
    os.remove("Pareto Front.html")

    radar(
        moop1,
        db='pf',
        points='none',
        output='html')
    assert (os.path.exists("Pareto Front.html"))
    os.remove("Pareto Front.html")

    # * obj x none x constraints in MOOP
    scatter(
        moop1,
        db='obj',
        points='none',
        output='html')
    assert (os.path.exists("Objective Data.html"))
    os.remove("Objective Data.html")

    parallel_coordinates(
        moop1,
        db='obj',
        points='none',
        output='html')
    assert (os.path.exists("Objective Data.html"))
    os.remove("Objective Data.html")

    radar(
        moop1,
        db='obj',
        points='none',
        output='html')
    assert (os.path.exists("Objective Data.html"))
    os.remove("Objective Data.html")

    # * pf x constraint_satisfying x no constraints in MOOP
    scatter(
        moop2,
        db='pf',
        points='constraint_satisfying',
        output='html')
    assert (os.path.exists("Pareto Front.html"))
    os.remove("Pareto Front.html")

    parallel_coordinates(
        moop2,
        db='pf',
        points='constraint_satisfying',
        output='html')
    assert (os.path.exists("Pareto Front.html"))
    os.remove("Pareto Front.html")

    radar(
        moop2,
        db='pf',
        points='constraint_satisfying',
        output='html')
    assert (os.path.exists("Pareto Front.html"))
    os.remove("Pareto Front.html")

    # * obj x constraint_satisfying x no constraints in MOOP
    scatter(
        moop2,
        db='obj',
        points='constraint_satisfying',
        output='html')
    assert (os.path.exists("Objective Data.html"))
    os.remove("Objective Data.html")

    parallel_coordinates(
        moop2,
        db='obj',
        points='constraint_satisfying',
        output='html')
    assert (os.path.exists("Objective Data.html"))
    os.remove("Objective Data.html")

    radar(
        moop2,
        db='obj',
        points='constraint_satisfying',
        output='html')
    assert (os.path.exists("Objective Data.html"))
    os.remove("Objective Data.html")

    # * pf x constraint_violating x no constraints in MOOP
    scatter(
        moop2,
        db='pf',
        points='constraint_violating',
        output='html')
    assert (os.path.exists("Pareto Front.html"))
    os.remove("Pareto Front.html")

    parallel_coordinates(
        moop2,
        db='pf',
        points='constraint_violating',
        output='html')
    assert (os.path.exists("Pareto Front.html"))
    os.remove("Pareto Front.html")

    radar(
        moop2,
        db='pf',
        points='constraint_violating',
        output='html')
    assert (os.path.exists("Pareto Front.html"))
    os.remove("Pareto Front.html")

    # * obj x constraint_violating x no constraints in MOOP
    scatter(
        moop2,
        db='obj',
        points='constraint_violating',
        output='html')
    assert (os.path.exists("Objective Data.html"))
    os.remove("Objective Data.html")

    parallel_coordinates(
        moop2,
        db='obj',
        points='constraint_violating',
        output='html')
    assert (os.path.exists("Objective Data.html"))
    os.remove("Objective Data.html")

    radar(
        moop2,
        db='obj',
        points='constraint_violating',
        output='html')
    assert (os.path.exists("Objective Data.html"))
    os.remove("Objective Data.html")

    # * pf x all x no constraints in MOOP
    scatter(
        moop2,
        db='pf',
        points='all',
        output='html')
    assert (os.path.exists("Pareto Front.html"))
    os.remove("Pareto Front.html")

    parallel_coordinates(
        moop2,
        db='pf',
        points='all',
        output='html')
    assert (os.path.exists("Pareto Front.html"))
    os.remove("Pareto Front.html")

    radar(
        moop2,
        db='pf',
        points='all',
        output='html')
    assert (os.path.exists("Pareto Front.html"))
    os.remove("Pareto Front.html")

    # * obj x all x no constraints in MOOP
    scatter(
        moop2,
        db='obj',
        points='all',
        output='html')
    assert (os.path.exists("Objective Data.html"))
    os.remove("Objective Data.html")

    parallel_coordinates(
        moop2,
        db='obj',
        points='all',
        output='html')
    assert (os.path.exists("Objective Data.html"))
    os.remove("Objective Data.html")

    radar(
        moop2,
        db='obj',
        points='all',
        output='html')
    assert (os.path.exists("Objective Data.html"))
    os.remove("Objective Data.html")

    # * pf x none x no constraints in MOOP
    scatter(
        moop2,
        db='pf',
        points='none',
        output='html')
    assert (os.path.exists("Pareto Front.html"))
    os.remove("Pareto Front.html")

    parallel_coordinates(
        moop2,
        db='pf',
        points='none',
        output='html')
    assert (os.path.exists("Pareto Front.html"))
    os.remove("Pareto Front.html")

    radar(
        moop2,
        db='pf',
        points='none',
        output='html')
    assert (os.path.exists("Pareto Front.html"))
    os.remove("Pareto Front.html")

    # * obj x none x no constraints in MOOP
    scatter(
        moop2,
        db='obj',
        points='none',
        output='html')
    assert (os.path.exists("Objective Data.html"))
    os.remove("Objective Data.html")

    parallel_coordinates(
        moop2,
        db='obj',
        points='none',
        output='html')
    assert (os.path.exists("Objective Data.html"))
    os.remove("Objective Data.html")

    radar(
        moop2,
        db='obj',
        points='none',
        output='html')
    assert (os.path.exists("Objective Data.html"))
    os.remove("Objective Data.html")


def test_inputs_to_dash():
    """ Stress-test the dash app's error handling. """

    from parmoo.viz.plot import scatter
    import os
    import pytest

    # Pre-calculate a moop object
    moop1 = run_quickstart()

    # * db
    # valid db values tested in test_database_options()
    # test invalid db values
    with pytest.raises(ValueError):
        scatter(moop1, output='html', db='asdf')

    with pytest.raises(ValueError):
        scatter(moop1, output='html', db=1234)

    # * output
    # valid output values tested in test_static_export()
    # test invalid output values
    with pytest.raises(ValueError):
        scatter(moop1, output='asdf')

    with pytest.raises(ValueError):
        scatter(moop1, output=1234)

    # * points
    # valid points values tested in test_database_options()
    # test invalid points values
    with pytest.raises(ValueError):
        scatter(moop1, output='html', points='asdf')

    with pytest.raises(ValueError):
        scatter(moop1, output='html', points=1234)

    # * height
    # test valid height values
    scatter(moop1, output='html', height=1)
    assert (os.path.exists("Pareto Front.html"))
    os.remove("Pareto Front.html")

    scatter(moop1, output='html', height=50)
    assert (os.path.exists("Pareto Front.html"))
    os.remove("Pareto Front.html")

    scatter(moop1, output='html', height=92.2678)
    assert (os.path.exists("Pareto Front.html"))
    os.remove("Pareto Front.html")

    scatter(moop1, output='html', height=7789)
    assert (os.path.exists("Pareto Front.html"))
    os.remove("Pareto Front.html")

    # test invalid height values
    with pytest.raises(ValueError):
        scatter(moop1, output='html', height='asdf')

    # * width
    # test valid width values
    scatter(moop1, output='html', width=1)
    assert (os.path.exists("Pareto Front.html"))
    os.remove("Pareto Front.html")

    scatter(moop1, output='html', width=50)
    assert (os.path.exists("Pareto Front.html"))
    os.remove("Pareto Front.html")

    scatter(moop1, output='html', width=92.2678)
    assert (os.path.exists("Pareto Front.html"))
    os.remove("Pareto Front.html")

    scatter(moop1, output='html', width=7789)
    assert (os.path.exists("Pareto Front.html"))
    os.remove("Pareto Front.html")

    # test invalid width values
    with pytest.raises(ValueError):
        scatter(moop1, output='html', width='asdf')

    # * font
    # test valid font values
    scatter(moop1, output='html', font='Verdana')
    assert (os.path.exists("Pareto Front.html"))
    os.remove("Pareto Front.html")

    scatter(moop1, output='html', font='Times New Roman')
    assert (os.path.exists("Pareto Front.html"))
    os.remove("Pareto Front.html")

    # test invalid font values
    # currently all inputs that can be cast to a string are valid
    # if str(font) != a font in the computer, font=Times New Roman

    # * fontsize
    # test valid fontsize values
    scatter(moop1, output='html', fontsize=1)
    assert (os.path.exists("Pareto Front.html"))
    os.remove("Pareto Front.html")

    scatter(moop1, output='html', fontsize=50)
    assert (os.path.exists("Pareto Front.html"))
    os.remove("Pareto Front.html")

    scatter(moop1, output='html', fontsize=55.71)
    assert (os.path.exists("Pareto Front.html"))
    os.remove("Pareto Front.html")

    scatter(moop1, output='html', fontsize=100)
    assert (os.path.exists("Pareto Front.html"))
    os.remove("Pareto Front.html")

    # test invalid fontsize values
    with pytest.raises(ValueError):
        scatter(moop1, output='html', fontsize=-1)

    with pytest.raises(ValueError):
        scatter(moop1, output='html', fontsize=101)

    # * background_color
    # test valid background_color values
    scatter(moop1, output='html', background_color='white')
    assert (os.path.exists("Pareto Front.html"))
    os.remove("Pareto Front.html")

    scatter(moop1, output='html', background_color='black')
    assert (os.path.exists("Pareto Front.html"))
    os.remove("Pareto Front.html")

    scatter(
        moop1,
        output='html',
        background_color='transparent'
    )
    assert (os.path.exists("Pareto Front.html"))
    os.remove("Pareto Front.html")

    scatter(moop1, output='html', background_color='white')
    assert (os.path.exists("Pareto Front.html"))
    os.remove("Pareto Front.html")

    scatter(moop1, output='html', background_color='grey')
    assert (os.path.exists("Pareto Front.html"))
    os.remove("Pareto Front.html")

    # test invalid background_color values
    # currently all inputs that can be cast to a string are valid
    # if str(background_color) != a supported background_color
    # background_color=white

    # * screenshot
    # test valid screenshot values
    scatter(moop1, output='html', screenshot='png')
    assert (os.path.exists("Pareto Front.html"))
    os.remove("Pareto Front.html")

    scatter(moop1, output='html', screenshot='jpeg')
    assert (os.path.exists("Pareto Front.html"))
    os.remove("Pareto Front.html")

    scatter(moop1, output='html', screenshot='svg')
    assert (os.path.exists("Pareto Front.html"))
    os.remove("Pareto Front.html")

    scatter(moop1, output='html', screenshot='webp')
    assert (os.path.exists("Pareto Front.html"))
    os.remove("Pareto Front.html")

    # test invalid screenshot values
    with pytest.raises(ValueError):
        scatter(moop1, output='html', screenshot='asdf')

    with pytest.raises(ValueError):
        scatter(moop1, output='html', screenshot=1234)

    # * image_export_format
    # test valid image_export_format values (except eps)
    scatter(moop1, output='html', image_export_format='html')
    assert (os.path.exists("Pareto Front.html"))
    os.remove("Pareto Front.html")

    scatter(moop1, output='html', image_export_format='svg')
    assert (os.path.exists("Pareto Front.html"))
    os.remove("Pareto Front.html")

    scatter(moop1, output='html', image_export_format='pdf')
    assert (os.path.exists("Pareto Front.html"))
    os.remove("Pareto Front.html")

    scatter(moop1, output='html', image_export_format='png')
    assert (os.path.exists("Pareto Front.html"))
    os.remove("Pareto Front.html")

    scatter(moop1, output='html', image_export_format='jpeg')
    assert (os.path.exists("Pareto Front.html"))
    os.remove("Pareto Front.html")

    scatter(moop1, output='html', image_export_format='webp')
    assert (os.path.exists("Pareto Front.html"))
    os.remove("Pareto Front.html")

    # test invalid image_export_format values
    with pytest.raises(ValueError):
        scatter(moop1, output='html', image_export_format='asdf')

    with pytest.raises(ValueError):
        scatter(moop1, output='html', image_export_format=1234)

    # * data_export_format
    # test valid data_export_format values
    scatter(moop1, output='html', data_export_format='csv')
    assert (os.path.exists("Pareto Front.html"))
    os.remove("Pareto Front.html")

    scatter(moop1, output='html', data_export_format='json')
    assert (os.path.exists("Pareto Front.html"))
    os.remove("Pareto Front.html")

    # test invalid data_export_format values
    with pytest.raises(ValueError):
        scatter(moop1, output='html', data_export_format='asdf')

    with pytest.raises(ValueError):
        scatter(moop1, output='html', data_export_format=1234)

    # * dev_mode
    # test valid dev_mode values
    scatter(moop1, output='html', dev_mode=True)
    assert (os.path.exists("Pareto Front.html"))
    os.remove("Pareto Front.html")

    scatter(moop1, output='html', dev_mode=False)
    assert (os.path.exists("Pareto Front.html"))
    os.remove("Pareto Front.html")

    # test invalid dev_mode values
    # currently all inputs that can be cast to a boolean are valid
    # in Python, everything can be cast to a boolean

    # * pop_up
    # test valid pop_up values
    scatter(moop1, output='html', pop_up=True)
    assert (os.path.exists("Pareto Front.html"))
    os.remove("Pareto Front.html")

    scatter(moop1, output='html', pop_up=False)
    assert (os.path.exists("Pareto Front.html"))
    os.remove("Pareto Front.html")

    # test invalid pop_up values
    # currently all inputs that can be cast to a boolean are valid
    # in Python, everything can be cast to a boolean

    # * port
    # test valid port values
    scatter(
        moop1,
        output='html',
        port='http://127.0.0.1:8050/'
    )
    assert (os.path.exists("Pareto Front.html"))
    os.remove("Pareto Front.html")

    # test invalid port values
    with pytest.raises(ValueError):
        scatter(moop1, output='html', port='asdf')

    with pytest.raises(ValueError):
        scatter(moop1, output='html', port=1234)


def run_quickstart():
    """ Auxiliary function that creates a MOOP by running the quickstart. """

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

    for i in range(3):
        my_moop.addAcquisition({
            'acquisition': UniformWeights,
            'hyperparams': {}
        })

    my_moop.solve(1)

    return my_moop


def run_dtlz2():
    """ Auxiliary function that creates a MOOP by running DTLZ2. """

    from parmoo import MOOP
    from parmoo.acquisitions import RandomConstraint
    from parmoo.searches import LatinHypercube
    from parmoo.surrogates import GaussRBF
    from parmoo.optimizers import GlobalSurrogate_BFGS
    from parmoo.objectives.dtlz import dtlz2_obj, dtlz2_grad
    from parmoo.simulations.dtlz import g2_sim

    n = 6  # number of design variables
    o = 5  # number of objectives
    q = 4  # batch size (number of acquisitions)
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
    test_static_export()
    test_quantity_constraints_objectives()
    test_database_options()
    test_inputs_to_dash()
