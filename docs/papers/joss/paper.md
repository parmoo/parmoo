---
title: 'ParMOO: A Python library for parallel multiobjective simulation optimization'
tags:
  - Python
  - numerical optimization
  - multiobjective optimization 
  - response surface methodology
  - parallel simulations
authors:
  - name: Tyler H. Chang^[Corresponding author]
    orcid: 0000-0001-9541-7041
    affiliation: "1" 
  - name: Stefan M. Wild
    orcid: 0000-0002-6099-2772
    affiliation: "1, 2" # (Multiple affiliations must be quoted)    
affiliations:
 - name: Mathematics and Computer Science Division, Argonne National Laboratory, USA
   index: 1
 - name: NAISE, Northwestern University, USA 
   index: 2
date: 12 May 2022
bibliography: paper.bib

---

# Summary

A multiobjective optimization problem (MOOP) is an optimization problem
in which multiple objectives are optimized simultaneously.
The goal of a MOOP is to find solutions that describe the tradeoff
curve between these (potentially conflicting) objectives.
This tradeoff curve is called the Pareto front.
Real-world MOOPs may also involve constraints -- additional hard rules
that every solution must adhere to.
In a multiobjective *simulation* optimization problem, the objectives are
derived from the outputs of one or more computationally expensive simulations.
Such problems are ubiquitous in science and engineering.

ParMOO is a Python framework for solving multiobjective simulation
optimization problems in parallel and at scale, while exploiting structure
in how the simulation outputs are used to formulate the optimization problem.
In the *Example Problems* section, we will see two examples of such structure.
First, a case where the objectives are formed by taking the sum-of-squared
simulation outputs, and another where two objectives are obtained through an
expensive real-world experiment, but the third objective is just an algebraic
equation.

Additionally, ParMOO is:

 - an open-source project on [GitHub](https://github.com/parmoo/parmoo),
 - ``pip``-installable via [PyPI](https://pypi.org/project/parmoo), and
 - fully [documented](https://parmoo.readthedocs.io).

# Multiobjective Simulation Optimization Software

Existing open source, actively maintained Python packages for solving
generic multiobjective simulation optimization problems include
``pymoo`` @pymoo,
``pymoso`` @pymoso,
``Dragonfly`` @dragonfly,
``Playtpus``,
``jMetalPy``,
``pygmo``.
Non multi-objective optimization specific Python packages, that are
often used for implementing multiobjective optimization solvers include
``BoTorch``,
``DEAP``.
Other non-Python packages include
``PlatEMO``,
``jMetal``,
``MODIR`` @modir,
``VTMOP`` @vtmop, and
``BoostDFO`` @boostdfo.
And two older packages, that are still worth mentioning include:
``ParEGO`` and
``SPEA2``.

The above-listed software packages:

 a) are not restricted to a particular MOOP application,
 b) have source code publicly available for download,
 c) are suitable for or contain sub-modules for solving a general form of the
    multiobjective simulation optimization problem, and
 d) provide sufficient documentation for a new user to get started
    without requiring council from the authors.

# Statement of Need

All of the previously mentioned software packages are high-quality and/or
feature-complete in some sense.
However, ParMOO is distinct for the following reasons:

 1) ParMOO is designed to be flexible enough to support diverse scientific
    workflows and solve a wide variety of real-world problem types.
 2) ParMOO provides interfaces and solver techniques that are suitable for both
    introductory and expert users.
 3) ParMOO provides bells and whistles that are required in production-quality
    solvers, such as checkpointing and logging.
 4) By layering on top of libEnsemble [@libensemble], ParMOO provides an 
    easy-to-use interface for distributing expensive simulation calculations
    over HPC resources.
 5) ParMOO provides complete documentation, including instructions for
    potential contributors.
 6) ParMOO is designed around extensibility and continuous-integration, with
    the intention of adding support for new features, solvers, techniques,
    and problem types, some of which may be beyond what we originally
    envisioned.
 7) In situations where there is an exploitable simulation-based structure
    in the problem definition
    (e.g., one or more objectives/constraints does not depend on any
    simulation outputs or the objectives are sum-of-squares of simulation
    outputs),
    then ParMOO can exploit this structure by collecting simulation data
    and modeling simulation outputs independently from the objectives and
    constraints.

While many existing solvers provide one or more of properties 1-5,
at this time, no other solver has *all* of these properties at once.
Additionally, to our knowledge, properties 6 and 7 are unique to ParMOO.

# Example Problems

To demonstrate the utility of ParMOO and the importance of
property 7, we describe two current applications.

First, ParMOO is currently being used to calibrate energy density functional
(EDF) models, by minimizing the error between expensive simulation outputs and
experimental data.
In this context, the simulation-based structure comes from the known equation
of the empirical loss function, such as the sum-of-squared errors for
a particular class of simulation outputs.
This example also illustrates ParMOO's ability to utilize
parallel resources (property 4), since the expensive EDF simulations
are being distributed over HPC resources using libEnsemble.

Second, ParMOO is being used to automate material design and manufacturing
in a wet lab-based environment, where each "simulation evaluation"
corresponds to the experimental synthesis and characterization of a
particular material.
The simulation-based structure in this problem comes from the known dependence
between the computed objectives (such as the total material yield and
byproduct) and the raw experimental data, which is typically a distribution
of measurements.
This example also demonstrates how ParMOO is able to easily integrate with
the material scientists' tools and workflow (property 1), which could not
be wrapped in a simple Python simulation function.

# Acknowledgements

We would like to thank Jeffrey Larson, Stephen Hudson, and John-Luke Navarro
for their advice on documentation, automated testing, and package setup.

This work was supported in part by the U.S. Department of Energy, 
Office of Science, Office of Advanced Scientific Computing Research, 
Scientific Discovery through Advanced Computing (SciDAC) program 
through the FASTMath Institute under Contract No.\ DE-AC02-06CH11357.
This work was supported by the National Science Foundation CSSI 
program under award number OAC-2004601 (BAND Collaboration).

# References


