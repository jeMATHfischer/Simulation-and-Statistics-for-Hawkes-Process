#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 14:17:26 2018

@author: jens
"""

import plotly
import plotly.graph_objs as go
import itertools
import numpy as np
from Surface_plot_datagenerator import DataGenerator

params_a = np.linspace(0.1,1.9,10)
params_b = np.linspace(0.1,1.9,10)
z = np.array([DataGenerator(item) for item in itertools.product(params_a, params_b)])
z = np.reshape(z, (len(params_a),-1))    

np.savetxt('quot_a_{}_{}_b_{}_{}.txt'.format(str(params_a[0]).replace('.',''),str(params_a[-1]).replace('.',''),str(params_b[0]).replace('.',''),str(params_b[-1]).replace('.','')), z)


z = np.loadtxt('quot_a_{}_{}_b_{}_{}.txt'.format(str(params_a[0]).replace('.',''),str(params_a[-1]).replace('.',''),str(params_b[0]).replace('.',''),str(params_b[-1]).replace('.','')))

print(z)

data = [
    go.Surface(
        x = params_a,
        y = params_b,
        z=z
    )
]
layout = go.Layout(
    title='Parameter-surface',
    autosize=False,
    width=500,
    height=500,
    margin=dict(
        l=65,
        r=50,
        b=65,
        t=90
    )
)
    
    
fig = go.Figure(data=data, layout=layout)
plotly.offline.plot(fig, filename='Parameter_surface_quot.html')


trace = go.Heatmap(z=z,
                   x = params_b,
                   y = params_a,
                   colorscale = 'Viridis'
                   )

layout = go.Layout(
    title='Parameter-Space Heatmap',
    xaxis=dict(
        title='Parameter b',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    ),
    yaxis=dict(
        title='Parameter a',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    )
)

data_heat=[trace]

fig = go.Figure(data=data_heat, layout=layout)

plotly.offline.plot(fig, filename='Parameter_heatmap_quot.html')
