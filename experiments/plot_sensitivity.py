import matplotlib.pyplot as plt
import numpy as np
import pandas

import matplotlib.pyplot as plt
plt.style.use('ggplot')

from pandas.tools.plotting import parallel_coordinates
def parallel_coordinates(frame, class_column, cols=None, ax=None, color=None,
                     use_columns=False, xticks=None, colormap="viridis",
                     **kwds):
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    n = len(frame)
    class_col = frame[class_column]
    class_min = np.amin(class_col)
    class_max = np.amax(class_col)

    if cols is None:
        df = frame.drop(class_column, axis=1)
    else:
        df = frame[cols]

    used_legends = set([])

    ncols = len(df.columns)

    # determine values to use for xticks
    if use_columns is True:
        if not np.all(np.isreal(list(df.columns))):
            raise ValueError('Columns must be numeric to be used as xticks')
        x = df.columns
    elif xticks is not None:
        if not np.all(np.isreal(xticks)):
            raise ValueError('xticks specified must be numeric')
        elif len(xticks) != ncols:
            raise ValueError('Length of xticks must match number of columns')
        x = xticks
    else:
        x = range(ncols)

    fig = plt.figure()
    ax = plt.gca()

    Colorm = plt.get_cmap(colormap)

    for i in range(n):
        y = df.iloc[i].values
        kls = class_col.iat[i]
        ax.plot(x, y, color=Colorm((kls - class_min)/(class_max-class_min)), **kwds)

    for i in x:
        ax.axvline(i, linewidth=1, color='black')

    ax.set_xticks(x)
    ax.set_xticklabels(df.columns)
    ax.set_xlim(x[0], x[-1])
    ax.legend(loc='upper right')
    ax.grid()

    bounds = np.linspace(class_min,class_max,10)
    cax,_ = mpl.colorbar.make_axes(ax)
    cb = mpl.colorbar.ColorbarBase(cax, cmap=Colorm, spacing='proportional', ticks=bounds, boundaries=bounds, format='%.2f')

    return fig
from pandas import DataFrame

f = np.loadtxt("../sensitivity_out.txt", delimiter=",")
print(f)
f = f[:,np.array([True, True, False, True, True, True, True])]
f[:,0] = f[:,0] / 9.
f[:,2] = f[:,2] / 5.
f[:,3] = np.array(f[:,3] * 5.1, dtype=int) / 5.0
f[:,4] = np.array(f[:,4] * 5.1, dtype=int) / 5.0
f[:,0] += np.random.randn(*(f[:,0].shape)) * 0.005
f[:,1] += np.random.randn(*(f[:,0].shape)) * 0.005
f[:,2] += np.random.randn(*(f[:,0].shape)) * 0.005
f[:,3] += np.random.randn(*(f[:,0].shape)) * 0.005
f[:,4] += np.random.randn(*(f[:,0].shape)) * 0.005

f = f[f[:,-1].argsort()]
print(f)
md = DataFrame(f[:,:-1], columns=["Num Cars", "Traffic Lights", "Num Pedestrians", "Noise", "Omission Prob"])
print(f[:,-1].shape)
md['class'] = f[:,-1]

parallel_coordinates(md, 'class')

plt.show()
