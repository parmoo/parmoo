The Interactive Visualization (viz) Library
-------------------------------------------

Easily generate a locally-hosted GUI for interactively visualizing your MOOP
results.

The public interface can be accessed by importing viz module, which contains
three functions for creating graphically interactive visualizations of the
Pareto front and objective data, which run on the browser in a Dash app.

.. code-block:: python

    import parmoo.viz

Public Plotting Functions Running in Python Plotly and Dash
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: viz.plot
..    :members: viz/plot

.. autofunction:: scatter
.. autofunction:: parallel_coordinates
.. autofunction:: radar

Other Private Classes and Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are several other private submodules, classes, and functions contained
in the viz module, that are used for creating and hosting the dashboard.
These do not need to be referenced by the user, but are detailed below
for developers.

Please note that documentation for some of these functions may be incomplete.

.. automodule:: viz.dashboard
..    :members: viz/dashboard

.. autoclass:: Dash_App
   :member-order: bysource
   :members:

   .. automethod:: __init__
   .. automethod:: generate_graph
   .. automethod:: configure
   .. automethod:: update_height
   .. automethod:: update_width
   .. automethod:: update_font
   .. automethod:: update_font_size
   .. automethod:: update_plot_name
   .. automethod:: update_background_color
   .. automethod:: update_plot_type
   .. automethod:: update_database
   .. automethod:: evaluate_height
   .. automethod:: evaluate_width
   .. automethod:: evaluate_font
   .. automethod:: evaluate_font_size
   .. automethod:: evaluate_background_color
   .. automethod:: evaluate_plot_name
   .. automethod:: evaluate_plot_type
   .. automethod:: evaluate_database
   .. automethod:: evaluate_dataset_download
   .. automethod:: evaluate_selected_data
   .. automethod:: set_constraint_range
   .. automethod:: update_constraint_range
   .. automethod:: evaluate_selection_download
   .. automethod:: evaluate_image_export_format
   .. automethod:: evaluate_data_export_format
   .. automethod:: evaluate_image_download
   .. automethod:: evaluate_customization_options
   .. automethod:: evaluate_export_options
   .. automethod:: evaluate_constraint_showr

.. automodule:: viz.graph
..    :members: viz/graph

.. autofunction:: generate_scatter
.. autofunction:: generate_parallel
.. autofunction:: generate_radar

.. automodule:: viz.utilities
..    :members: viz/utilities

.. autofunction:: export_file
.. autofunction:: set_plot_name
.. autofunction:: set_database
.. autofunction:: set_hover_info
.. autofunction:: check_inputs
