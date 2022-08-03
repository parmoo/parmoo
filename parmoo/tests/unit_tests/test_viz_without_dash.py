def test_static_export():

    # tests all export formats except eps
    # (because eps may not be supported on all machines)

    from parmoo.viz.plot import (
        scatter,
        parallel_coordinates,
        radar,
    )
    import os

    # * html output
    scatter(moop=run_quickstart(), output='html')
    assert(os.path.exists("Pareto front.html"))
    os.remove("Pareto front.html")

    # * svg output
    parallel_coordinates(moop=run_quickstart(), output='svg')
    assert(os.path.exists("Pareto front.svg"))
    os.remove("Pareto front.svg")

    # * pdf output
    radar(moop=run_quickstart(), output='pdf')
    assert(os.path.exists("Pareto front.pdf"))
    os.remove("Pareto front.pdf")

    # * jpeg output
    scatter(moop=run_quickstart(), output='jpeg')
    assert(os.path.exists("Pareto front.jpeg"))
    os.remove("Pareto front.jpeg")

    # * png output
    parallel_coordinates(moop=run_quickstart(), output='png')
    assert(os.path.exists("Pareto front.png"))
    os.remove("Pareto front.png")

    # * webp output
    radar(moop=run_quickstart(), output='webp')
    assert(os.path.exists("Pareto front.webp"))
    os.remove("Pareto front.webp")


def test_quantity_constraints_objectives():

    from parmoo.viz.plot import (
        scatter,
        parallel_coordinates,
        radar,
    )
    import os

    # * 2 objective scatter with constraint
    scatter(moop=run_quickstart(), output='html')
    assert(os.path.exists("Pareto front.html"))
    os.remove("Pareto front.html")

    # * 2 objective parallel with constraint
    parallel_coordinates(moop=run_quickstart(), output='html')
    assert(os.path.exists("Pareto front.html"))
    os.remove("Pareto front.html")

    # * 2 objective radar with constraint
    radar(moop=run_quickstart(), output='html')
    assert(os.path.exists("Pareto front.html"))
    os.remove("Pareto front.html")

    # * 5 objective scatter without constraint
    scatter(moop=run_dtlz2(), output='html')
    assert(os.path.exists("Pareto front.html"))
    os.remove("Pareto front.html")

    # * 5 objective parallel without constraint
    parallel_coordinates(moop=run_dtlz2(), output='html')
    assert(os.path.exists("Pareto front.html"))
    os.remove("Pareto front.html")

    # * 5 objective radar without constraint
    radar(moop=run_dtlz2(), output='html')
    assert(os.path.exists("Pareto front.html"))
    os.remove("Pareto front.html")


def test_database_options():

    # test all database-plotting options for all plots

    from parmoo.viz.plot import (
        scatter,
        parallel_coordinates,
        radar,
    )
    import os

    # * pf x constraint_satisfying x constraints in MOOP
    scatter(
        moop=run_quickstart(),
        db='pf',
        points='constraint_satisfying',
        output='html')
    assert(os.path.exists("Pareto front.html"))
    os.remove("Pareto front.html")

    parallel_coordinates(
        moop=run_quickstart(),
        db='pf',
        points='constraint_satisfying',
        output='html')
    assert(os.path.exists("Pareto front.html"))
    os.remove("Pareto front.html")

    radar(
        moop=run_quickstart(),
        db='pf',
        points='constraint_satisfying',
        output='html')
    assert(os.path.exists("Pareto front.html"))
    os.remove("Pareto front.html")

    # * obj x constraint_satisfying x constraints in MOOP
    scatter(
        moop=run_quickstart(),
        db='obj',
        points='constraint_satisfying',
        output='html')
    assert(os.path.exists("Objective data.html"))
    os.remove("Objective data.html")

    parallel_coordinates(
        moop=run_quickstart(),
        db='obj',
        points='constraint_satisfying',
        output='html')
    assert(os.path.exists("Objective data.html"))
    os.remove("Objective data.html")

    radar(
        moop=run_quickstart(),
        db='obj',
        points='constraint_satisfying',
        output='html')
    assert(os.path.exists("Objective data.html"))
    os.remove("Objective data.html")

    # * pf x constraint_violating x constraints in MOOP
    scatter(
        moop=run_quickstart(),
        db='pf',
        points='constraint_violating',
        output='html')
    assert(os.path.exists("Pareto front.html"))
    os.remove("Pareto front.html")

    parallel_coordinates(
        moop=run_quickstart(),
        db='pf',
        points='constraint_violating',
        output='html')
    assert(os.path.exists("Pareto front.html"))
    os.remove("Pareto front.html")

    radar(
        moop=run_quickstart(),
        db='pf',
        points='constraint_violating',
        output='html')
    assert(os.path.exists("Pareto front.html"))
    os.remove("Pareto front.html")

    # * obj x constraint_violating x constraints in MOOP
    scatter(
        moop=run_quickstart(),
        db='obj',
        points='constraint_violating',
        output='html')
    assert(os.path.exists("Objective data.html"))
    os.remove("Objective data.html")

    parallel_coordinates(
        moop=run_quickstart(),
        db='obj',
        points='constraint_violating',
        output='html')
    assert(os.path.exists("Objective data.html"))
    os.remove("Objective data.html")

    radar(
        moop=run_quickstart(),
        db='obj',
        points='constraint_violating',
        output='html')
    assert(os.path.exists("Objective data.html"))
    os.remove("Objective data.html")

    # * pf x all x constraints in MOOP
    scatter(
        moop=run_quickstart(),
        db='pf',
        points='all',
        output='html')
    assert(os.path.exists("Pareto front.html"))
    os.remove("Pareto front.html")

    parallel_coordinates(
        moop=run_quickstart(),
        db='pf',
        points='all',
        output='html')
    assert(os.path.exists("Pareto front.html"))
    os.remove("Pareto front.html")

    radar(
        moop=run_quickstart(),
        db='pf',
        points='all',
        output='html')
    assert(os.path.exists("Pareto front.html"))
    os.remove("Pareto front.html")

    # * obj x all x constraints in MOOP
    scatter(
        moop=run_quickstart(),
        db='obj',
        points='all',
        output='html')
    assert(os.path.exists("Objective data.html"))
    os.remove("Objective data.html")

    parallel_coordinates(
        moop=run_quickstart(),
        db='obj',
        points='all',
        output='html')
    assert(os.path.exists("Objective data.html"))
    os.remove("Objective data.html")

    radar(
        moop=run_quickstart(),
        db='obj',
        points='all',
        output='html')
    assert(os.path.exists("Objective data.html"))
    os.remove("Objective data.html")

    # * pf x none x constraints in MOOP
    scatter(
        moop=run_quickstart(),
        db='pf',
        points='none',
        output='html')
    assert(os.path.exists("Pareto front.html"))
    os.remove("Pareto front.html")

    parallel_coordinates(
        moop=run_quickstart(),
        db='pf',
        points='none',
        output='html')
    assert(os.path.exists("Pareto front.html"))
    os.remove("Pareto front.html")

    radar(
        moop=run_quickstart(),
        db='pf',
        points='none',
        output='html')
    assert(os.path.exists("Pareto front.html"))
    os.remove("Pareto front.html")

    # * obj x none x constraints in MOOP
    scatter(
        moop=run_quickstart(),
        db='obj',
        points='none',
        output='html')
    assert(os.path.exists("Objective data.html"))
    os.remove("Objective data.html")

    parallel_coordinates(
        moop=run_quickstart(),
        db='obj',
        points='none',
        output='html')
    assert(os.path.exists("Objective data.html"))
    os.remove("Objective data.html")

    radar(
        moop=run_quickstart(),
        db='obj',
        points='none',
        output='html')
    assert(os.path.exists("Objective data.html"))
    os.remove("Objective data.html")

    # * pf x constraint_satisfying x no constraints in MOOP
    scatter(
        moop=run_dtlz2(),
        db='pf',
        points='constraint_satisfying',
        output='html')
    assert(os.path.exists("Pareto front.html"))
    os.remove("Pareto front.html")

    parallel_coordinates(
        moop=run_dtlz2(),
        db='pf',
        points='constraint_satisfying',
        output='html')
    assert(os.path.exists("Pareto front.html"))
    os.remove("Pareto front.html")

    radar(
        moop=run_dtlz2(),
        db='pf',
        points='constraint_satisfying',
        output='html')
    assert(os.path.exists("Pareto front.html"))
    os.remove("Pareto front.html")

    # * obj x constraint_satisfying x no constraints in MOOP
    scatter(
        moop=run_dtlz2(),
        db='obj',
        points='constraint_satisfying',
        output='html')
    assert(os.path.exists("Objective data.html"))
    os.remove("Objective data.html")

    parallel_coordinates(
        moop=run_dtlz2(),
        db='obj',
        points='constraint_satisfying',
        output='html')
    assert(os.path.exists("Objective data.html"))
    os.remove("Objective data.html")

    radar(
        moop=run_dtlz2(),
        db='obj',
        points='constraint_satisfying',
        output='html')
    assert(os.path.exists("Objective data.html"))
    os.remove("Objective data.html")

    # * pf x constraint_violating x no constraints in MOOP
    scatter(
        moop=run_dtlz2(),
        db='pf',
        points='constraint_violating',
        output='html')
    assert(os.path.exists("Pareto front.html"))
    os.remove("Pareto front.html")

    parallel_coordinates(
        moop=run_dtlz2(),
        db='pf',
        points='constraint_violating',
        output='html')
    assert(os.path.exists("Pareto front.html"))
    os.remove("Pareto front.html")

    radar(
        moop=run_dtlz2(),
        db='pf',
        points='constraint_violating',
        output='html')
    assert(os.path.exists("Pareto front.html"))
    os.remove("Pareto front.html")

    # * obj x constraint_violating x no constraints in MOOP
    scatter(
        moop=run_dtlz2(),
        db='obj',
        points='constraint_violating',
        output='html')
    assert(os.path.exists("Objective data.html"))
    os.remove("Objective data.html")

    parallel_coordinates(
        moop=run_dtlz2(),
        db='obj',
        points='constraint_violating',
        output='html')
    assert(os.path.exists("Objective data.html"))
    os.remove("Objective data.html")

    radar(
        moop=run_dtlz2(),
        db='obj',
        points='constraint_violating',
        output='html')
    assert(os.path.exists("Objective data.html"))
    os.remove("Objective data.html")

    # * pf x all x no constraints in MOOP
    scatter(
        moop=run_dtlz2(),
        db='pf',
        points='all',
        output='html')
    assert(os.path.exists("Pareto front.html"))
    os.remove("Pareto front.html")

    parallel_coordinates(
        moop=run_dtlz2(),
        db='pf',
        points='all',
        output='html')
    assert(os.path.exists("Pareto front.html"))
    os.remove("Pareto front.html")

    radar(
        moop=run_dtlz2(),
        db='pf',
        points='all',
        output='html')
    assert(os.path.exists("Pareto front.html"))
    os.remove("Pareto front.html")

    # * obj x all x no constraints in MOOP
    scatter(
        moop=run_dtlz2(),
        db='obj',
        points='all',
        output='html')
    assert(os.path.exists("Objective data.html"))
    os.remove("Objective data.html")

    parallel_coordinates(
        moop=run_dtlz2(),
        db='obj',
        points='all',
        output='html')
    assert(os.path.exists("Objective data.html"))
    os.remove("Objective data.html")

    radar(
        moop=run_dtlz2(),
        db='obj',
        points='all',
        output='html')
    assert(os.path.exists("Objective data.html"))
    os.remove("Objective data.html")

    # * pf x none x no constraints in MOOP
    scatter(
        moop=run_dtlz2(),
        db='pf',
        points='none',
        output='html')
    assert(os.path.exists("Pareto front.html"))
    os.remove("Pareto front.html")

    parallel_coordinates(
        moop=run_dtlz2(),
        db='pf',
        points='none',
        output='html')
    assert(os.path.exists("Pareto front.html"))
    os.remove("Pareto front.html")

    radar(
        moop=run_dtlz2(),
        db='pf',
        points='none',
        output='html')
    assert(os.path.exists("Pareto front.html"))
    os.remove("Pareto front.html")

    # * obj x none x no constraints in MOOP
    scatter(
        moop=run_dtlz2(),
        db='obj',
        points='none',
        output='html')
    assert(os.path.exists("Objective data.html"))
    os.remove("Objective data.html")

    parallel_coordinates(
        moop=run_dtlz2(),
        db='obj',
        points='none',
        output='html')
    assert(os.path.exists("Objective data.html"))
    os.remove("Objective data.html")

    radar(
        moop=run_dtlz2(),
        db='obj',
        points='none',
        output='html')
    assert(os.path.exists("Objective data.html"))
    os.remove("Objective data.html")


def test_inputs_to_dash():

    from parmoo.viz.plot import (
        scatter,
        parallel_coordinates,
        radar,
    )
    import os
    import pytest

    # * db
    # valid db values tested in test_database_options()
    # test invalid db values
    with pytest.raises(ValueError):
        scatter(moop=run_quickstart(), output='html', db='asdf')

    with pytest.raises(ValueError):
        scatter(moop=run_quickstart(), output='html', db=1234)

    # * output
    # valid output values tested in test_static_export()
    # test invalid output values
    with pytest.raises(ValueError):
        scatter(moop=run_quickstart(), output='asdf')

    with pytest.raises(ValueError):
        scatter(moop=run_quickstart(), output=1234)

    # * points
    # valid points values tested in test_database_options()
    # test invalid points values
    with pytest.raises(ValueError):
        scatter(moop=run_quickstart(), output='html', points='asdf')

    with pytest.raises(ValueError):
        scatter(moop=run_quickstart(), output='html', points=1234)

    # * height
    # test valid height values
    scatter(moop=run_quickstart(), output='html', height=1)
    assert(os.path.exists("Pareto front.html"))
    os.remove("Pareto front.html")

    scatter(moop=run_quickstart(), output='html', height=50)
    assert(os.path.exists("Pareto front.html"))
    os.remove("Pareto front.html")

    scatter(moop=run_quickstart(), output='html', height=92.2678)
    assert(os.path.exists("Pareto front.html"))
    os.remove("Pareto front.html")

    scatter(moop=run_quickstart(), output='html', height=7789)
    assert(os.path.exists("Pareto front.html"))
    os.remove("Pareto front.html")

    # test invalid height values
    with pytest.raises(ValueError):
        scatter(moop=run_quickstart(), output='html', height='asdf')

    # * width
    # test valid width values
    scatter(moop=run_quickstart(), output='html', width=1)
    assert(os.path.exists("Pareto front.html"))
    os.remove("Pareto front.html")

    scatter(moop=run_quickstart(), output='html', width=50)
    assert(os.path.exists("Pareto front.html"))
    os.remove("Pareto front.html")

    scatter(moop=run_quickstart(), output='html', width=92.2678)
    assert(os.path.exists("Pareto front.html"))
    os.remove("Pareto front.html")

    scatter(moop=run_quickstart(), output='html', width=7789)
    assert(os.path.exists("Pareto front.html"))
    os.remove("Pareto front.html")

    # test invalid width values
    with pytest.raises(ValueError):
        scatter(moop=run_quickstart(), output='html', width='asdf')

    # * font
    # test valid font values
    scatter(moop=run_quickstart(), output='html', font='Verdana')
    assert(os.path.exists("Pareto front.html"))
    os.remove("Pareto front.html")

    scatter(moop=run_quickstart(), output='html', font='Times New Roman')
    assert(os.path.exists("Pareto front.html"))
    os.remove("Pareto front.html")

    # test invalid font values
    # currently all inputs that can be cast to a string are valid
    # if str(font) != a font in the computer, font=Times New Roman

    # * fontsize
    # test valid fontsize values
    scatter(moop=run_quickstart(), output='html', fontsize=1)
    assert(os.path.exists("Pareto front.html"))
    os.remove("Pareto front.html")

    scatter(moop=run_quickstart(), output='html', fontsize=50)
    assert(os.path.exists("Pareto front.html"))
    os.remove("Pareto front.html")

    scatter(moop=run_quickstart(), output='html', fontsize=55.71)
    assert(os.path.exists("Pareto front.html"))
    os.remove("Pareto front.html")

    scatter(moop=run_quickstart(), output='html', fontsize=100)
    assert(os.path.exists("Pareto front.html"))
    os.remove("Pareto front.html")

    # test invalid fontsize values
    with pytest.raises(ValueError):
        scatter(moop=run_quickstart(), output='html', fontsize=-1)

    with pytest.raises(ValueError):
        scatter(moop=run_quickstart(), output='html', fontsize=101)

    # * background_color
    # test valid background_color values
    scatter(moop=run_quickstart(), output='html', background_color='white')
    assert(os.path.exists("Pareto front.html"))
    os.remove("Pareto front.html")

    scatter(moop=run_quickstart(), output='html', background_color='black')
    assert(os.path.exists("Pareto front.html"))
    os.remove("Pareto front.html")

    scatter(
        moop=run_quickstart(),
        output='html',
        background_color='transparent'
    )
    assert(os.path.exists("Pareto front.html"))
    os.remove("Pareto front.html")

    scatter(moop=run_quickstart(), output='html', background_color='white')
    assert(os.path.exists("Pareto front.html"))
    os.remove("Pareto front.html")

    scatter(moop=run_quickstart(), output='html', background_color='grey')
    assert(os.path.exists("Pareto front.html"))
    os.remove("Pareto front.html")

    # test invalid background_color values
    # currently all inputs that can be cast to a string are valid
    # if str(background_color) != a supported background_color
    # background_color=white

    # * screenshot
    # test valid screenshot values
    scatter(moop=run_quickstart(), output='html', screenshot='png')
    assert(os.path.exists("Pareto front.html"))
    os.remove("Pareto front.html")

    scatter(moop=run_quickstart(), output='html', screenshot='jpeg')
    assert(os.path.exists("Pareto front.html"))
    os.remove("Pareto front.html")

    scatter(moop=run_quickstart(), output='html', screenshot='svg')
    assert(os.path.exists("Pareto front.html"))
    os.remove("Pareto front.html")

    scatter(moop=run_quickstart(), output='html', screenshot='webp')
    assert(os.path.exists("Pareto front.html"))
    os.remove("Pareto front.html")

    # test invalid screenshot values
    with pytest.raises(ValueError):
        scatter(moop=run_quickstart(), output='html', screenshot='asdf')

    with pytest.raises(ValueError):
        scatter(moop=run_quickstart(), output='html', screenshot=1234)

    # * image_export_format
    # test valid image_export_format values (except eps)
    scatter(moop=run_quickstart(), output='html', image_export_format='html')
    assert(os.path.exists("Pareto front.html"))
    os.remove("Pareto front.html")

    scatter(moop=run_quickstart(), output='html', image_export_format='svg')
    assert(os.path.exists("Pareto front.html"))
    os.remove("Pareto front.html")

    scatter(moop=run_quickstart(), output='html', image_export_format='pdf')
    assert(os.path.exists("Pareto front.html"))
    os.remove("Pareto front.html")

    scatter(moop=run_quickstart(), output='html', image_export_format='png')
    assert(os.path.exists("Pareto front.html"))
    os.remove("Pareto front.html")

    scatter(moop=run_quickstart(), output='html', image_export_format='jpeg')
    assert(os.path.exists("Pareto front.html"))
    os.remove("Pareto front.html")

    scatter(moop=run_quickstart(), output='html', image_export_format='webp')
    assert(os.path.exists("Pareto front.html"))
    os.remove("Pareto front.html")

    # test invalid image_export_format values
    with pytest.raises(ValueError):
        scatter(moop=run_quickstart(), output='html', image_export_format='asdf')

    with pytest.raises(ValueError):
        scatter(moop=run_quickstart(), output='html', image_export_format=1234)

    # * data_export_format
    # test valid data_export_format values
    scatter(moop=run_quickstart(), output='html', data_export_format='csv')
    assert(os.path.exists("Pareto front.html"))
    os.remove("Pareto front.html")

    scatter(moop=run_quickstart(), output='html', data_export_format='json')
    assert(os.path.exists("Pareto front.html"))
    os.remove("Pareto front.html")

    # test invalid data_export_format values
    with pytest.raises(ValueError):
        scatter(moop=run_quickstart(), output='html', data_export_format='asdf')

    with pytest.raises(ValueError):
        scatter(moop=run_quickstart(), output='html', data_export_format=1234)

    # * dev_mode
    # test valid dev_mode values
    scatter(moop=run_quickstart(), output='html', dev_mode=True)
    assert(os.path.exists("Pareto front.html"))
    os.remove("Pareto front.html")

    scatter(moop=run_quickstart(), output='html', dev_mode=False)
    assert(os.path.exists("Pareto front.html"))
    os.remove("Pareto front.html")

    # test invalid dev_mode values
    # currently all inputs that can be cast to a boolean are valid
    # in Python, everything can be cast to a boolean

    # * pop_up
    # test valid pop_up values
    scatter(moop=run_quickstart(), output='html', pop_up=True)
    assert(os.path.exists("Pareto front.html"))
    os.remove("Pareto front.html")

    scatter(moop=run_quickstart(), output='html', pop_up=False)
    assert(os.path.exists("Pareto front.html"))
    os.remove("Pareto front.html")

    # test invalid pop_up values
    # currently all inputs that can be cast to a boolean are valid
    # in Python, everything can be cast to a boolean

    # * port
    # test valid port values
    scatter(
        moop=run_quickstart(),
        output='html',
        port='http://127.0.0.1:8050/'
    )
    assert(os.path.exists("Pareto front.html"))
    os.remove("Pareto front.html")

    # test invalid port values
    with pytest.raises(ValueError):
        scatter(moop=run_quickstart(), output='html', port='asdf')

    with pytest.raises(ValueError):
        scatter(moop=run_quickstart(), output='html', port=1234)


def run_quickstart():

    import numpy as np
    from parmoo import MOOP
    from parmoo.searches import LatinHypercube
    from parmoo.surrogates import GaussRBF
    from parmoo.acquisitions import UniformWeights
    from parmoo.optimizers import LocalGPS

    my_moop = MOOP(LocalGPS)

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
        'hyperparams': {'search_budget': 20}
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

    my_moop.solve(5)

    return my_moop


def run_dtlz2():
    from parmoo import MOOP
    from parmoo.acquisitions import RandomConstraint
    from parmoo.searches import LatinHypercube
    from parmoo.surrogates import GaussRBF
    from parmoo.optimizers import LBFGSB
    from parmoo.objectives.dtlz import dtlz2_obj
    from parmoo.simulations.dtlz import g2_sim

    n = 6  # number of design variables
    o = 5  # number of objectives
    q = 4  # batch size (number of acquisitions)

    # Create MOOP
    moop = MOOP(LBFGSB)
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
        'hyperparams': {'search_budget': 10*n}
    })
    # Add o objectives
    for i in range(o):
        moop.addObjective({
            'name': f"Objective {i+1}",
            'obj_func': dtlz2_obj(
                moop.getDesignType(),
                moop.getSimulationType(),
                i, num_obj=o)
        })

    # Add q acquisition functions
    for i in range(q):
        moop.addAcquisition({'acquisition': RandomConstraint})
    # Solve the MOOP with 20 iterations
    moop.solve(5)

    return moop


if __name__ == "__main__":
    test_static_export()
    test_quantity_constraints_objectives()
    test_database_options()
    test_inputs_to_dash()
