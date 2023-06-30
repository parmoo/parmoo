FAQ
===

Frequently asked questions:

 - Q: Why are there so many hyperparameters?

    - A: ParMOO is **not a solver** it is a
      **framework for building and deploying customized solvers**.
      It has been our experience that a single solver cannot obtain the
      best performance on every problem.
      Through ParMOO, we are giving you a modeling language and library
      of useful techniques for building the perfect solver for you problem
      and deploying it in parallel environments.

 - Q: There are too many options, where do I start?

    - A: If you are not an optimization expert or if you're just not
      sure where to start, that's OK.
      Start from one of our tutorials_ and modify it to match your problem.
      See how that works for your problem.
      Then slowly start experimenting with other options, refer to this
      FAQ for some general guidance.
      Alternatively, check out some of our existing pre-built solvers in
      the parmoo_solver_farm_ and see if you can modify one to fit your
      needs!

 - Q: I am running ParMOO with a single simulation function with
   the following settings: ``search_budget = B_s``,
   ``number of acquisition functions = q``,
   ``max_iters (input to solve method) = k``.
   How many times will ParMOO run my simulation while solving this problem?

    - A: In this example: ``total_budget = B_s + q * k``

 - Q: Going off the previous example, how should I choose the values of
   ``B_s``, ``q``, and ``k``?

    - A: The answer is problem dependent, but these are our recommendations:

       - For most of our solver settings (there are several notable
         exceptions, such as our Bayesian optimization implementations)
         ParMOO performs pure "exploitation" after the initial search,
         so you can think of ``B_s`` as the "exploration" parameter.
         If your problem is very
         smooth or doesn't have many local minima, then you can get away with
         a small value of ``B_s``. If your problem is nonsmooth or highly
         nonconvex, you will do better with a larger value of ``B_s``. At a
         minimum, ``B_s`` must be at least ``n+1`` (recall: ``n`` is the
         number of design variables). All of our existing surrogate models
         will raise an error if ``B_s < n+1``.
         If you are still unsure, a good starting place that works well on
         a wide variety of problems is ``B_s = min(2000, 0.5*total_budget)``,
         where ``total_budget`` is the total budget that you intend to use
         for this problem.
       - The value of ``q`` determines how many simulations ParMOO will
         evaluate per batch. If you want to achieve NP-way parallelism (using
         the ``libE_MOOP`` class), then you should use ``q=NP``. If you are
         running serially (using the base ``MOOP`` class) then you should use
         a small number of acquisition functions in order to get more
         iterations.
         In many situations, just 1 acquisition function will work best when
         running serially. In order to amortize the cost of re-fitting the
         surrogate model, we typically we use ``q=2``, ``3``, or ``4`` when
         running serially.
       - The value of ``k`` should be as large as you can afford, given your
         simulation costs. Note that the ``GaussRBF`` surrogate becomes
         expensive to fit for ``total_budget > 4000``, although we have used
         it for up to ``total_budget = 10,000``. If you want more practical
         advice, see the next question.

 - Q: Given the advice from the previous question, how do I know a good
   ``total_budget`` for my problem?

    - A: It depends on a lot of factors. In general, if your problem is
      small (``n < 8`` design variables **and** ``o < 3`` objectives) you can
      probably get away with a total budget in the **hundreds**.
      If your problem is large (``n > 8`` design variables **or**
      ``o > 3`` objectives) you will probably need a total budget in the
      **thousands** or even **ten thousands**.
      In general, as you increase the number of design variables or objectives
      the problem expense increases **exponentially** if you want to maintain
      the same accuracy. This is part of the *curse of dimensionality*.

 - Q: I have a lot of design variables but I can't afford that large of a
   budget, what can I do?

    - A: The key issue is that **global optimization is expensive**.
      At a fundamental level, we cannot guarantee global convergence without
      densly sampling the design space, which is exponentially expensive
      when ``n`` (number of design variables) is large.
      So what can you do?
      You can switch to using **local modeling methods**, whose costs
      generally only grow linearly in the dimension.
      You will not get any global convergence guarantees, but in many
      cases, you will still be able to solve your problem.

 - Q: How can I determine whether my problem was solved by ParMOO?

    - Short Answer: you can't. Here's why -- in most situations (unless
      your objectives are not truly conflicting) there are infinitely
      many solutions to a multiobjective optimization problem. We cannot
      find all of them on a finite budget. What we can do is give you as
      many approximate solutions as possible for the budget allocated, so
      that you can make informed decisions about the inherent tradeoffs,
      and possibly run a single-objective solver to refine your favorite
      solution in the future.
    - Long Answers:

       - For practical purposes: you could solve the problem with ParMOO on
         as large of a budget as you can afford with checkpointing turned
         on. Then plot the results using one of the methods from our
         :mod:`viz <viz>`
         library and see how you are doing. If you are un-satisfied with the
         results, re-load from the last checkpoint and solve with a few added
         iterations. Then plot your results again and see if the performance
         has improved.
       - For small problems: you could solve the problem with ParMOO on
         as large of a budget as you can afford. Then plot the convergence
         over time, according to one of the common multiobjective performance
         indicators, such as hypervolume. If you are seeing diminishing
         improvements in late iterations, then it is likely that you have
         solved the problem. Note that hypervolume is exponentially expensive
         to compute when you have a large number of objectives. Therefore, we
         do not have a hypervolume metric calculator available in ParMOO at
         this time, but we will add it in the future.

 - Q: Surrogate models, acquisition functions, search techniques, and
   optimization solvers -- how do I know which ones to pick?

    - A: Generally, we recommend sticking with
      :class:`LatinHypercube <searches.latin_hypercube.LatinHypercube>`
      search and
      :class:`RandomConstraint <acquisitions.epsilon_constraint.RandomConstraint>`
      acquisition functions, unless you have a good reason for changing.
      These options work best for most of our test problems,
      and they are demonstrated in our tutorials.
      For the surrogate model and optimization solver, start out with
      :class:`LocalGPS <optimizers.gps_search.LocalGPS>` optimizer
      and :class:`GaussRBF <surrogates.gaussian_proc.GaussRBF>` surrogate
      model, as in the quickstart_.
      Then:

       - If you are willing to code the derivative for your objective
         and constraint functions (not the simulations), then you can
         follow the advanced_example_ and switch to using the
         :class:`LBFGSB <optimizers.lbfgsb.LBFGSB>` optimizer.
       - If you have a lot of design variables, then you might do better
         with a local solver, by switching your surrogate to the
         :class:`LocalGaussRBF <surrogates.gaussian_proc.LocalGaussRBF>`
         surrogate.
         If you are using the
         :class:`LBFGSB <optimizers.lbfgsb.LBFGSB>` optimizer, then you
         will also need to switch to the
         :class:`TR_LBFGSB <optimizers.lbfgsb.TR_LBFGSB>` optimizer.
       - If you're a professional optimizer or researcher and you want
         to try your own methods, then you can do so by writing your own
         implementation for one of our
         :mod:`Abstract Base Classes <structs>`.
         If you try a novel method and it works and you're ready to publish
         it, consider sharing your novel solver on the
         parmoo_solver_farm_!


.. _advanced_example: https://parmoo.readthedocs.io/en/latest/tutorials/basic-tutorials.html#Solving
.. _parmoo_solver_farm: https://github.com/parmoo/parmoo-solver-farm
.. _quickstart: quickstart.html
.. _tutorials: tutorials/basic-tutorials.html
