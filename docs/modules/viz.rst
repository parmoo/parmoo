Interactive visualizations based on Python Plotly and Dash
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Easily generate a locally-hosted GUI for interactively visualizing your MOOP results.

.. code-block:: python

    import parmoo.viz

The viz module contains functions to create a Dash app containing graphically
customizable plots of the Pareto Front and objective data.

.. automodule:: viz.plot
..    :members: viz/plot

.. autofunction:: scatter
.. autofunction:: parallel_coordinates
.. autofunction:: radar

.. automodule:: viz.dashboard
..    :members: viz/dashboard

.. autoclass:: Dash_App
   :member-order: bysource
   :members:

   .. automethod:: __init__
   .. automethod:: generate_graph
   .. automethod:: configure
   .. automethod:: update_height
   .. automethod:: update_width,
   .. automethod:: update_font,
   .. automethod:: update_font_size,
   .. automethod:: update_plot_name,
   .. automethod:: update_background_color,
   .. automethod:: update_plot_type,
   .. automethod:: update_database,
   .. automethod:: evaluate_height,
   .. automethod:: evaluate_width,
   .. automethod:: evaluate_font,
   .. automethod:: evaluate_font_size,
   .. automethod:: evaluate_background_color,
   .. automethod:: evaluate_plot_name,
   .. automethod:: evaluate_plot_type,
   .. automethod:: evaluate_database,
   .. automethod:: evaluate_dataset_download,
   .. automethod:: evaluate_selected_data,
   .. automethod:: set_constraint_range,
   .. automethod:: update_constraint_range,
   .. automethod:: evaluate_selection_download,
   .. automethod:: evaluate_image_export_format,
   .. automethod:: evaluate_data_export_format,
   .. automethod:: evaluate_image_download,
   .. automethod:: evaluate_customization_options,
   .. automethod:: evaluate_export_options,
   .. automethod:: evaluate_constraint_showr,

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
