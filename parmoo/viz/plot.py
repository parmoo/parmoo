import logging
from .dashboard import (
    Dash_App
)
from .utilities import (
    export_file,
    set_plot_name,
    check_inputs,
)
from .graph import (
    generate_scatter,
    generate_parallel,
    generate_radar,
)


def scatter(
    moop,
    db='pf',
    output='dash',
    points='constraint_satisfying',
    height='auto',
    width='auto',
    font='auto',
    fontsize='auto',
    background_color='auto',
    screenshot='svg',
    image_export_format='svg',
    data_export_format='csv',
    dev_mode=False,
    pop_up=True,
    port='http://127.0.0.1:8050/',
):
    check_inputs(
        db=db,
        output=output,
        points=points,
        height=height,
        width=width,
        font=font,
        fontsize=fontsize,
        background_color=background_color,
        screenshot=screenshot,
        image_export_format=image_export_format,
        data_export_format=data_export_format,
        dev_mode=dev_mode,
        pop_up=pop_up,
        port=port,
    )

    logging.info('initialized scatter() wrapper')

    # * output
    if output == 'dash':
        Dash_App(
            plot_type='scatter',
            moop=moop,
            db=db,
            points=points,
            height=height,
            width=width,
            font=font,
            fontsize=fontsize,
            background_color=background_color,
            screenshot=screenshot,
            image_export_format=image_export_format,
            data_export_format=data_export_format,
            dev_mode=dev_mode,
            pop_up=pop_up,
            port=port,
        )
    else:
        plot_name = set_plot_name(db=db)
        export_file(
            fig=generate_scatter(
                moop,
                db=db,
                points=points,
            ),
            plot_name=plot_name,
            file_type=output,
        )


def parallel_coordinates(
    moop,
    db='pf',
    output='dash',
    points='constraint_satisfying',
    height='auto',
    width='auto',
    font='auto',
    fontsize='auto',
    background_color='auto',
    screenshot='svg',
    image_export_format='svg',
    data_export_format='csv',
    dev_mode=False,
    pop_up=True,
    port='http://127.0.0.1:8050/',
):
    check_inputs(
        db=db,
        output=output,
        points=points,
        height=height,
        width=width,
        font=font,
        fontsize=fontsize,
        background_color=background_color,
        screenshot=screenshot,
        image_export_format=image_export_format,
        data_export_format=data_export_format,
        dev_mode=dev_mode,
        pop_up=pop_up,
        port=port,
    )

    logging.info('initialized parallel_coordinates() wrapper')

    if output == 'dash':
        Dash_App(
            plot_type='parallel',
            moop=moop,
            db=db,
            points=points,
            height=height,
            width=width,
            font=font,
            fontsize=fontsize,
            background_color=background_color,
            screenshot=screenshot,
            image_export_format=image_export_format,
            data_export_format=data_export_format,
            dev_mode=dev_mode,
            pop_up=pop_up,
            port=port,
        )
    else:
        plot_name = set_plot_name(db=db)
        export_file(
            fig=generate_parallel(
                moop=moop,
                db=db,
                points=points,
            ),
            plot_name=plot_name,
            file_type=output
        )


def radar(
    moop,
    db='pf',
    output='dash',
    points='constraint_satisfying',
    height='auto',
    width='auto',
    font='auto',
    fontsize='auto',
    background_color='auto',
    screenshot='svg',
    image_export_format='svg',
    data_export_format='csv',
    dev_mode=False,
    pop_up=True,
    port='http://127.0.0.1:8050/',
):
    check_inputs(
        db=db,
        output=output,
        points=points,
        height=height,
        width=width,
        font=font,
        fontsize=fontsize,
        background_color=background_color,
        screenshot=screenshot,
        image_export_format=image_export_format,
        data_export_format=data_export_format,
        dev_mode=dev_mode,
        pop_up=pop_up,
        port=port,
    )
    logging.info('initialized radar() wrapper')
    if output == 'dash':
        Dash_App(
            plot_type='radar',
            moop=moop,
            db=db,
            points=points,
            height=height,
            width=width,
            font=font,
            fontsize=fontsize,
            background_color=background_color,
            screenshot=screenshot,
            image_export_format=image_export_format,
            data_export_format=data_export_format,
            dev_mode=dev_mode,
            pop_up=pop_up,
            port=port,
        )
    else:
        plot_name = set_plot_name(db=db)
        export_file(
            fig=generate_radar(
                moop,
                db=db,
                points=points,
            ),
            plot_name=plot_name,
            file_type=output
        )
