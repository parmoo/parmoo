from .dashboard import *


def scatter(moop,
            db='pf',
            output='dash',
            height='auto',
            width='auto',
            verbose=True,
            hot_reload=False,
            pop_up=True,
            port='http://127.0.0.1:8050/',):
    if output == 'dash':
        buildDashApp(moop=moop,
                 plotType='scatter',
                 db=db,
                 height='auto',
                 width='auto',
                 verbose=verbose,
                 hot_reload=hot_reload,
                 pop_up=pop_up,
                 port=port,)
    # elif output == 'no_dash':
    #     graph.scatter(config=config)
    else:
        exportFile(fig=fig,
                   plotName=plotName,
                   fileType=output)


def radar(moop,
          db='pf',
          output='dash',
          height='auto',
          width='auto',
          verbose=True,
          hot_reload=False,
          pop_up=True,
          port='http://127.0.0.1:8050/',):

    if output == 'dash':
        buildDashApp(moop=moop,
                 plotType='radar',
                 db=db,
                 height=height,
                 width=width,
                 verbose=verbose,
                 hot_reload=hot_reload,
                 pop_up=pop_up,
                 port=port,)
    # elif output == 'no_dash':
    #     graph.scatter(config=config)
    else:
        exportFile(fig=fig,
                   plotName=plotName,
                   fileType=output)

def parallel_coordinates(moop,
                         db='pf',
                         output='dash',
                         height='auto',
                         width='auto',
                         objectives_only=True,
                         verbose=True,
                         hot_reload=False,
                         pop_up=True,
                         port='http://127.0.0.1:8050/',):

    if output == 'dash':
        buildDashApp(moop=moop,
                 plotType='parallel_coordinates',
                 db=db,
                 height=height,
                 width=width,
                 verbose=verbose,
                 hot_reload=hot_reload,
                 pop_up=pop_up,
                 port=port,)
    # elif output == 'no_dash':
    #     graph.scatter(config=config)
    else:
        exportFile(fig=fig,
                   plotName=plotName,
                   fileType=output)

