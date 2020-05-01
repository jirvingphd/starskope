






############  SYMMETRY PLOTS ############ 
# ==========> https://docs.bokeh.org/en/latest/docs/user_guide/plotting.html

# Configure for Jupyter Notebook Display


from bokeh.plotting import figure
from bokeh.io import output_notebook, push_notebook, show

output_notebook()

plot = figure()
plot.circle([1,2,3], [4,6,5])

handle = show(plot, notebook_handle=True)

# Update the plot title in the earlier cell
plot.title.text = "New Title"
push_notebook(handle=handle)

### OUTPUT NOTEBOOK
output_notebook(notebook_type='jupyter')


#Configure output to a standalone HTML file.
### OUTBOOK FILE (HTML)
output_file(filename, title='Bokeh Plot', mode=None, root_dir=None)[source]


property document
A default Document to use for all output operations.

property file
A dict with the default configuration for file output (READ ONLY)

The dictionary value has the following form:

{
    'filename'  : # filename to use when saving
    'resources' : # resources configuration
    'title'     : # a title for the HTML document
}





from math import pi

import pandas as pd

from bokeh.io import show
from bokeh.models import BasicTicker, ColorBar, LinearColorMapper, PrintfTickFormatter
from bokeh.plotting import figure
from bokeh.sampledata.unemployment1948 import data

data['Year'] = data['Year'].astype(str)
data = data.set_index('Year')
data.drop('Annual', axis=1, inplace=True)
data.columns.name = 'Month'

years = list(data.index)
months = list(data.columns)

# reshape to 1D array or rates with a month and year for each row.
df = pd.DataFrame(data.stack(), columns=['rate']).reset_index()

# this is the colormap from the original NYTimes plot
colors = ["#75968f", "#a5bab7", "#c9d9d3", "#e2e2e2", "#dfccce", "#ddb7b1", "#cc7878", "#933b41", "#550b1d"]
mapper = LinearColorMapper(palette=colors, low=df.rate.min(), high=df.rate.max())

TOOLS = "hover,save,pan,box_zoom,reset,wheel_zoom"

p = figure(title="US Unemployment ({0} - {1})".format(years[0], years[-1]),
           x_range=years, y_range=list(reversed(months)),
           x_axis_location="above", plot_width=900, plot_height=400,
           tools=TOOLS, toolbar_location='below',
           tooltips=[('date', '@Month @Year'), ('rate', '@rate%')])

p.grid.grid_line_color = None
p.axis.axis_line_color = None
p.axis.major_tick_line_color = None
p.axis.major_label_text_font_size = "5pt"
p.axis.major_label_standoff = 0
p.xaxis.major_label_orientation = pi / 3

p.rect(x="Year", y="Month", width=1, height=1,
       source=df,
       fill_color={'field': 'rate', 'transform': mapper},
       line_color=None)

color_bar = ColorBar(color_mapper=mapper, major_label_text_font_size="5pt",
                     ticker=BasicTicker(desired_num_ticks=len(colors)),
                     formatter=PrintfTickFormatter(format="%d%%"),
                     label_standoff=6, border_line_color=None, location=(0, 0))
p.add_layout(color_bar, 'right')

show(p)      # show the plot



#### MARKERS

from numpy.random import random

from bokeh.plotting import figure, output_file, show


def mscatter(p, x, y, marker):
    p.scatter(x, y, marker=marker, size=15,
              line_color="navy", fill_color="orange", alpha=0.5)

def mtext(p, x, y, text):
    p.text(x, y, text=[text],
           text_color="firebrick", text_align="center", text_font_size="10pt")

p = figure(title="Bokeh Markers", toolbar_location=None)
p.grid.grid_line_color = None
p.background_fill_color = "#eeeeee"
p.axis.visible = False

N = 10

mscatter(p, random(N)+2, random(N)+1, "circle")
mscatter(p, random(N)+4, random(N)+1, "square")
mscatter(p, random(N)+6, random(N)+1, "triangle")
mscatter(p, random(N)+8, random(N)+1, "asterisk")

mscatter(p, random(N)+2, random(N)+4, "circle_x")
mscatter(p, random(N)+4, random(N)+4, "square_x")
mscatter(p, random(N)+6, random(N)+4, "inverted_triangle")
mscatter(p, random(N)+8, random(N)+4, "x")

mscatter(p, random(N)+2, random(N)+7, "circle_cross")
mscatter(p, random(N)+4, random(N)+7, "square_cross")
mscatter(p, random(N)+6, random(N)+7, "diamond")
mscatter(p, random(N)+8, random(N)+7, "cross")

mtext(p, 2.5, 0.5, "circle / o")
mtext(p, 4.5, 0.5, "square")
mtext(p, 6.5, 0.5, "triangle")
mtext(p, 8.5, 0.5, "asterisk / *")

mtext(p, 2.5, 3.5, "circle_x / ox")
mtext(p, 4.5, 3.5, "square_x")
mtext(p, 6.5, 3.5, "inverted_triangle")
mtext(p, 8.5, 3.5, "x")

mtext(p, 2.5, 6.5, "circle_cross / o+")
mtext(p, 4.5, 6.5, "square_cross")
mtext(p, 6.5, 6.5, "diamond")
mtext(p, 8.5, 6.5, "cross / +")

output_file("markers.html", title="markers.py example")

show(p)  # open a browser


# ========== HEXTILES
# Bokeh can plot hexagonal tiles, which are often used for showing binned aggregations. 
# The hex_tile() method takes a size parameter to define the size of the hex grid, 
# and axial coordinates to specify which tiles are present.

import numpy as np

from bokeh.io import output_file, show
from bokeh.plotting import figure
from bokeh.util.hex import axial_to_cartesian

output_file("hex_coords.html")

q = np.array([0,  0, 0, -1, -1,  1, 1])
r = np.array([0, -1, 1,  0,  1, -1, 0])

p = figure(plot_width=400, plot_height=400, toolbar_location=None)
p.grid.visible = False

p.hex_tile(q, r, size=1, fill_color=["firebrick"]*3 + ["navy"]*4,
           line_color="white", alpha=0.5)

x, y = axial_to_cartesian(q, r, 1, "pointytop")

p.text(x, y, text=["(%d, %d)" % (q,r) for (q, r) in zip(q, r)],
       text_baseline="middle", text_align="center")

show(p)


# ==========  


# computes counts per bin using the hexbin() function and plots the colormapped counts:
import numpy as np

from bokeh.io import output_file, show
from bokeh.plotting import figure
from bokeh.transform import linear_cmap
from bokeh.util.hex import hexbin

n = 50000
x = np.random.standard_normal(n)
y = np.random.standard_normal(n)

bins = hexbin(x, y, 0.1)

p = figure(tools="wheel_zoom,reset", match_aspect=True, background_fill_color='#440154')
p.grid.visible = False

p.hex_tile(q="q", r="r", size=0.1, line_color=None, source=bins,
           fill_color=linear_cmap('counts', 'Viridis256', 0, max(bins.counts)))

output_file("hex_tile.html")

show(p)




import numpy as np

from bokeh.io import output_file, show
from bokeh.models import HoverTool
from bokeh.plotting import figure

n = 500
x = 2 + 2*np.random.standard_normal(n)
y = 2 + 2*np.random.standard_normal(n)

p = figure(title="Hexbin for 500 points", match_aspect=True,
           tools="wheel_zoom,reset", background_fill_color='#440154')
p.grid.visible = False

r, bins = p.hexbin(x, y, size=0.5, hover_color="pink", hover_alpha=0.8)

p.circle(x, y, color="white", size=1)

p.add_tools(HoverTool(
    tooltips=[("count", "@c"), ("(q,r)", "(@q, @r)")],
    mode="mouse", point_policy="follow_mouse", renderers=[r]
))

output_file("hexbin.html")

show(p)


###### LINE SEGMENTS - GROUPS


from bokeh.plotting import figure, output_file, show

plot = figure(plot_width=300, plot_height=300)
plot.segment(x0=[1, 2, 3], y0=[1, 2, 3],
             x1=[1, 2, 3], y1=[1.2, 2.5, 3.7],
             color="#F4A582", line_width=3)

show(plot)




##### SQUARE CROSSES
from bokeh.plotting import figure, output_file, show

plot = figure(plot_width=300, plot_height=300)
plot.square_cross(x=[1, 2, 3], y=[1, 2, 3], size=[10,20,25],
                  color="#7FC97F",fill_color=None, line_width=2)

show(plot)



#### TRIANGLES

from bokeh.plotting import figure, output_file, show

plot = figure(plot_width=300, plot_height=300)
plot.triangle(x=[1, 2, 3], y=[1, 2, 3], size=[10,20,25],
              color="#99D594", line_width=2)

show(plot)


### CIRCLE CROSS

circle_cross(x, y, size=4, angle=0.0, *, angle_units='rad', 
            fill_alpha=1.0, fill_color='gray', 
            line_alpha=1.0, line_cap='butt', line_color='black', 
            line_dash=[], line_dash_offset=0, line_join='bevel', 
            line_width=1, name=None, tags=[], **kwargs)[source]


from bokeh.plotting import figure, output_file, show

plot = figure(plot_width=300, plot_height=300)
plot.circle_cross(x=[1,2,3], y=[4,5,6], size=20,
                  color="#FB8072", fill_alpha=0.2, line_width=2)

show(plot)


#### CIRCLE X

from bokeh.plotting import figure, output_file, show

plot = figure(plot_width=300, plot_height=300)
plot.circle_x(x=[1, 2, 3], y=[1, 2, 3], size=20,
              color="#DD1C77", fill_alpha=0.2)

show(plot)



# CROSS

from bokeh.plotting import figure, output_file, show

plot = figure(plot_width=300, plot_height=300)
plot.cross(x=[1, 2, 3], y=[1, 2, 3], size=20,
           color="#E6550D", line_width=2)

show(plot)



#### COLOR THEME: DARK MINIMAL

from bokeh.plotting import figure, output_file, show
from bokeh.themes import built_in_themes
from bokeh.io import curdoc

x = [1, 2, 3, 4, 5]
y = [6, 7, 6, 4, 5]

output_file("caliber.html")
curdoc().theme = 'caliber'
p = figure(title='caliber', plot_width=300, plot_height=300)
p.line(x, y)
show(p)


cosine(w: float, A: float = 1, phi: float = 0, offset: float = 0) → partial[Callable[[], None]][source]
Return a driver function that can advance a sequence of cosine values.

value = A * cos(w*i + phi) + offset
# Parameters
# w (float) – a frequency for the cosine driver

# A (float) – an amplitude for the cosine driver

# phi (float) – a phase offset to start the cosine driver with

# offset (float) – a global offset to add to the driver values 


#### SINE 

sine(w: float, A: float = 1, phi: float = 0, offset: float = 0) → partial[Callable[[], None]][source]¶
Return a driver function that can advance a sequence of sine values.

value = A * sin(w*i + phi) + offset
# Parameters
# w (float) – a frequency for the sine driver

# A (float) – an amplitude for the sine driver

# phi (float) – a phase offset to start the sine driver with

# offset (float) – a global offset to add to the driver values

# Project
# Roadmap
# Team
# Citation
# Documentation
# User Guide
# Gallery
# Reference Guide


# SINE WAVE
https://demo.bokeh.org/sliders


https://docs.bokeh.org/en/latest/docs/gallery/slider.html

import numpy as np

from bokeh.layouts import column, row
from bokeh.models import CustomJS, Slider
from bokeh.plotting import ColumnDataSource, figure, output_file, show

x = np.linspace(0, 10, 500)
y = np.sin(x)

source = ColumnDataSource(data=dict(x=x, y=y))

plot = figure(y_range=(-10, 10), plot_width=400, plot_height=400)

plot.line('x', 'y', source=source, line_width=3, line_alpha=0.6)

amp_slider = Slider(start=0.1, end=10, value=1, step=.1, title="Amplitude")
freq_slider = Slider(start=0.1, end=10, value=1, step=.1, title="Frequency")
phase_slider = Slider(start=0, end=6.4, value=0, step=.1, title="Phase")
offset_slider = Slider(start=-5, end=5, value=0, step=.1, title="Offset")

callback = CustomJS(args=dict(source=source, amp=amp_slider, freq=freq_slider, phase=phase_slider, offset=offset_slider),
                    code="""
    const data = source.data;
    const A = amp.value;
    const k = freq.value;
    const phi = phase.value;
    const B = offset.value;
    const x = data['x']
    const y = data['y']
    for (var i = 0; i < x.length; i++) {
        y[i] = B + A*Math.sin(k*x[i]+phi);
    }
    source.change.emit();
""")

amp_slider.js_on_change('value', callback)
freq_slider.js_on_change('value', callback)
phase_slider.js_on_change('value', callback)
offset_slider.js_on_change('value', callback)

layout = row(
    plot,
    column(amp_slider, freq_slider, phase_slider, offset_slider),
)

output_file("slider.html", title="slider.py example")

show(layout)


import numpy as np

from bokeh.layouts import gridplot
from bokeh.plotting import figure, output_file, show

x = np.linspace(0, 4*np.pi, 100)
y = np.sin(x)

TOOLS = "pan,wheel_zoom,box_zoom,reset,save,box_select"

p1 = figure(title="Legend Example", tools=TOOLS)

p1.circle(x,   y, legend_label="sin(x)")
p1.circle(x, 2*y, legend_label="2*sin(x)", color="orange")
p1.circle(x, 3*y, legend_label="3*sin(x)", color="green")

p1.legend.title = 'Example Title'

p2 = figure(title="Another Legend Example", tools=TOOLS)

p2.circle(x, y, legend_label="sin(x)")
p2.line(x, y, legend_label="sin(x)")

p2.line(x, 2*y, legend_label="2*sin(x)", line_dash=(4, 4), line_color="orange", line_width=2)

p2.square(x, 3*y, legend_label="3*sin(x)", fill_color=None, line_color="green")
p2.line(x, 3*y, legend_label="3*sin(x)", line_color="green")

output_file("legend.html", title="legend.py example")

show(gridplot([p1, p2], ncols=2, plot_width=400, plot_height=400))  # open a browser



https://attractors.pyviz.demo.anaconda.com/attractors_panel

https://demo.bokeh.org/stocks

# Interactive Weather Statistics DEMO
https://demo.bokeh.org/weather



https://docs.bokeh.org/en/latest/docs/gallery/burtin.html

from collections import OrderedDict
from io import StringIO
from math import log, sqrt

import numpy as np
import pandas as pd

from bokeh.plotting import figure, output_file, show

antibiotics = """
bacteria,                        penicillin, streptomycin, neomycin, gram
Mycobacterium tuberculosis,      800,        5,            2,        negative
Salmonella schottmuelleri,       10,         0.8,          0.09,     negative
Proteus vulgaris,                3,          0.1,          0.1,      negative
Klebsiella pneumoniae,           850,        1.2,          1,        negative
Brucella abortus,                1,          2,            0.02,     negative
Pseudomonas aeruginosa,          850,        2,            0.4,      negative
Escherichia coli,                100,        0.4,          0.1,      negative
Salmonella (Eberthella) typhosa, 1,          0.4,          0.008,    negative
Aerobacter aerogenes,            870,        1,            1.6,      negative
Brucella antracis,               0.001,      0.01,         0.007,    positive
Streptococcus fecalis,           1,          1,            0.1,      positive
Staphylococcus aureus,           0.03,       0.03,         0.001,    positive
Staphylococcus albus,            0.007,      0.1,          0.001,    positive
Streptococcus hemolyticus,       0.001,      14,           10,       positive
Streptococcus viridans,          0.005,      10,           40,       positive
Diplococcus pneumoniae,          0.005,      11,           10,       positive
"""

drug_color = OrderedDict([
    ("Penicillin",   "#0d3362"),
    ("Streptomycin", "#c64737"),
    ("Neomycin",     "black"  ),
])

gram_color = OrderedDict([
    ("negative", "#e69584"),
    ("positive", "#aeaeb8"),
])

df = pd.read_csv(StringIO(antibiotics),
                 skiprows=1,
                 skipinitialspace=True,
                 engine='python')

width = 800
height = 800
inner_radius = 90
outer_radius = 300 - 10

minr = sqrt(log(.001 * 1E4))
maxr = sqrt(log(1000 * 1E4))
a = (outer_radius - inner_radius) / (minr - maxr)
b = inner_radius - a * maxr

def rad(mic):
    return a * np.sqrt(np.log(mic * 1E4)) + b

big_angle = 2.0 * np.pi / (len(df) + 1)
small_angle = big_angle / 7

p = figure(plot_width=width, plot_height=height, title="",
    x_axis_type=None, y_axis_type=None,
    x_range=(-420, 420), y_range=(-420, 420),
    min_border=0, outline_line_color="black",
    background_fill_color="#f0e1d2")

p.xgrid.grid_line_color = None
p.ygrid.grid_line_color = None

# annular wedges
angles = np.pi/2 - big_angle/2 - df.index.to_series()*big_angle
colors = [gram_color[gram] for gram in df.gram]
p.annular_wedge(
    0, 0, inner_radius, outer_radius, -big_angle+angles, angles, color=colors,
)

# small wedges
p.annular_wedge(0, 0, inner_radius, rad(df.penicillin),
                -big_angle+angles+5*small_angle, -big_angle+angles+6*small_angle,
                color=drug_color['Penicillin'])
p.annular_wedge(0, 0, inner_radius, rad(df.streptomycin),
                -big_angle+angles+3*small_angle, -big_angle+angles+4*small_angle,
                color=drug_color['Streptomycin'])
p.annular_wedge(0, 0, inner_radius, rad(df.neomycin),
                -big_angle+angles+1*small_angle, -big_angle+angles+2*small_angle,
                color=drug_color['Neomycin'])

# circular axes and lables
labels = np.power(10.0, np.arange(-3, 4))
radii = a * np.sqrt(np.log(labels * 1E4)) + b
p.circle(0, 0, radius=radii, fill_color=None, line_color="white")
p.text(0, radii[:-1], [str(r) for r in labels[:-1]],
       text_font_size="8pt", text_align="center", text_baseline="middle")

# radial axes
p.annular_wedge(0, 0, inner_radius-10, outer_radius+10,
                -big_angle+angles, -big_angle+angles, color="black")

# bacteria labels
xr = radii[0]*np.cos(np.array(-big_angle/2 + angles))
yr = radii[0]*np.sin(np.array(-big_angle/2 + angles))
label_angle=np.array(-big_angle/2+angles)
label_angle[label_angle < -np.pi/2] += np.pi # easier to read labels on the left side
p.text(xr, yr, df.bacteria, angle=label_angle,
       text_font_size="9pt", text_align="center", text_baseline="middle")

# OK, these hand drawn legends are pretty clunky, will be improved in future release
p.circle([-40, -40], [-370, -390], color=list(gram_color.values()), radius=5)
p.text([-30, -30], [-370, -390], text=["Gram-" + gr for gr in gram_color.keys()],
       text_font_size="7pt", text_align="left", text_baseline="middle")

p.rect([-40, -40, -40], [18, 0, -18], width=30, height=13,
       color=list(drug_color.values()))
p.text([-15, -15, -15], [18, 0, -18], text=list(drug_color),
       text_font_size="9pt", text_align="left", text_baseline="middle")

output_file("burtin.html", title="burtin.py example")

show(p)


import colorcet as cc
from numpy import linspace
from scipy.stats.kde import gaussian_kde

from bokeh.io import output_file, show
from bokeh.models import ColumnDataSource, FixedTicker, PrintfTickFormatter
from bokeh.plotting import figure
from bokeh.sampledata.perceptions import probly

output_file("ridgeplot.html")

def ridge(category, data, scale=20):
    return list(zip([category]*len(data), scale*data))

cats = list(reversed(probly.keys()))

palette = [cc.rainbow[i*15] for i in range(17)]

x = linspace(-20,110, 500)

source = ColumnDataSource(data=dict(x=x))

p = figure(y_range=cats, plot_width=900, x_range=(-5, 105), toolbar_location=None)

for i, cat in enumerate(reversed(cats)):
    pdf = gaussian_kde(probly[cat])
    y = ridge(cat, pdf(x))
    source.add(y, cat)
    p.patch('x', cat, color=palette[i], alpha=0.6, line_color="black", source=source)

p.outline_line_color = None
p.background_fill_color = "#efefef"

p.xaxis.ticker = FixedTicker(ticks=list(range(0, 101, 10)))
p.xaxis.formatter = PrintfTickFormatter(format="%d%%")

p.ygrid.grid_line_color = None
p.xgrid.grid_line_color = "#dddddd"
p.xgrid.ticker = p.xaxis.ticker

p.axis.minor_tick_line_color = None
p.axis.major_tick_line_color = None
p.axis.axis_line_color = None

p.y_range.range_padding = 0.12

show(p)

### ANSCOMBES QUARTET

import numpy as np
import pandas as pd

from bokeh.document import Document
from bokeh.embed import file_html
from bokeh.layouts import column, gridplot
from bokeh.models import (Circle, ColumnDataSource, Div, Grid,
                          Line, LinearAxis, Plot, Range1d,)
from bokeh.resources import INLINE
from bokeh.util.browser import view

raw_columns=[
[10.0,   8.04,   10.0,   9.14,   10.0,   7.46,   8.0,    6.58],
[8.0,    6.95,   8.0,    8.14,   8.0,    6.77,   8.0,    5.76],
[13.0,   7.58,   13.0,   8.74,   13.0,   12.74,  8.0,    7.71],
[9.0,    8.81,   9.0,    8.77,   9.0,    7.11,   8.0,    8.84],
[11.0,   8.33,   11.0,   9.26,   11.0,   7.81,   8.0,    8.47],
[14.0,   9.96,   14.0,   8.10,   14.0,   8.84,   8.0,    7.04],
[6.0,    7.24,   6.0,    6.13,   6.0,    6.08,   8.0,    5.25],
[4.0,    4.26,   4.0,    3.10,   4.0,    5.39,   19.0,   12.5],
[12.0,   10.84,  12.0,   9.13,   12.0,   8.15,   8.0,    5.56],
[7.0,    4.82,   7.0,    7.26,   7.0,    6.42,   8.0,    7.91],
[5.0,    5.68,   5.0,    4.74,   5.0,    5.73,   8.0,    6.89]]

quartet = pd.DataFrame(data=raw_columns, columns=
                       ['Ix','Iy','IIx','IIy','IIIx','IIIy','IVx','IVy'])


circles_source = ColumnDataSource(
    data = dict(
        xi   = quartet['Ix'],
        yi   = quartet['Iy'],
        xii  = quartet['IIx'],
        yii  = quartet['IIy'],
        xiii = quartet['IIIx'],
        yiii = quartet['IIIy'],
        xiv  = quartet['IVx'],
        yiv  = quartet['IVy'],
    )
   )

x = np.linspace(-0.5, 20.5, 10)
y = 3 + 0.5 * x
lines_source = ColumnDataSource(data=dict(x=x, y=y))

xdr = Range1d(start=-0.5, end=20.5)
ydr = Range1d(start=-0.5, end=20.5)

def make_plot(title, xname, yname):
    plot = Plot(x_range=xdr, y_range=ydr, plot_width=400, plot_height=400,
                background_fill_color='#efefef')
    plot.title.text = title

    xaxis = LinearAxis(axis_line_color=None)
    plot.add_layout(xaxis, 'below')

    yaxis = LinearAxis(axis_line_color=None)
    plot.add_layout(yaxis, 'left')

    plot.add_layout(Grid(dimension=0, ticker=xaxis.ticker))
    plot.add_layout(Grid(dimension=1, ticker=yaxis.ticker))

    line = Line(x='x', y='y', line_color="#666699", line_width=2)
    plot.add_glyph(lines_source, line)

    circle = Circle(
        x=xname, y=yname, size=12,
        fill_color="#cc6633", line_color="#cc6633", fill_alpha=0.5
    )
    plot.add_glyph(circles_source, circle)

    return plot

#where will this comment show up
I   = make_plot('I',   'xi',   'yi')
II  = make_plot('II',  'xii',  'yii')
III = make_plot('III', 'xiii', 'yiii')
IV  = make_plot('IV',  'xiv',  'yiv')

grid = gridplot([[I, II], [III, IV]], toolbar_location=None)

div = Div(text="""
<h1>Anscombe's Quartet</h1>
<p>Anscombe's quartet is a collection of four small datasets that have nearly
identical simple descriptive statistics (mean, variance, correlation, and linear
regression lines), yet appear very different when graphed.
</p>
""")

doc = Document()
doc.add_root(column(div, grid, sizing_mode="scale_width"))

if __name__ == "__main__":
    doc.validate()
    filename = "anscombe.html"
    with open(filename, "w") as f:
        f.write(file_html(doc, INLINE, "Anscombe's Quartet"))
    print("Wrote %s" % filename)
    view(filename)



### CATEGORICAL DOT PLOT
from bokeh.layouts import row
from bokeh.plotting import figure, output_file, show

factors = ["a", "b", "c", "d", "e", "f", "g", "h"]
x =  [50, 40, 65, 10, 25, 37, 80, 60]

dot = figure(title="Categorical Dot Plot", tools="", toolbar_location=None,
            y_range=factors, x_range=[0,100])

dot.segment(0, factors, x, factors, line_width=2, line_color="green", )
dot.circle(x, factors, size=15, fill_color="orange", line_color="green", line_width=3, )

factors = ["foo 123", "bar:0.2", "baz-10"]
x = ["foo 123", "foo 123", "foo 123", "bar:0.2", "bar:0.2", "bar:0.2", "baz-10",  "baz-10",  "baz-10"]
y = ["foo 123", "bar:0.2", "baz-10",  "foo 123", "bar:0.2", "baz-10",  "foo 123", "bar:0.2", "baz-10"]
colors = [
    "#0B486B", "#79BD9A", "#CFF09E",
    "#79BD9A", "#0B486B", "#79BD9A",
    "#CFF09E", "#79BD9A", "#0B486B"
]

hm = figure(title="Categorical Heatmap", tools="hover", toolbar_location=None,
            x_range=factors, y_range=factors)

hm.rect(x, y, color=colors, width=1, height=1)

output_file("categorical.html", title="categorical.py example")

show(row(hm, dot, sizing_mode="scale_width"))  # open a browser



#### RGBA SQUARE

    import numpy as np

from bokeh.plotting import figure, output_file, show

N = 20
img = np.empty((N,N), dtype=np.uint32)
view = img.view(dtype=np.uint8).reshape((N, N, 4))
for i in range(N):
    for j in range(N):
        view[i, j, 0] = int(i/N*255)
        view[i, j, 1] = 158
        view[i, j, 2] = int(j/N*255)
        view[i, j, 3] = 255

p = figure(tooltips=[("x", "$x"), ("y", "$y"), ("value", "@image")])
p.x_range.range_padding = p.y_range.range_padding = 0

# must give a vector of images
p.image_rgba(image=[img], x=0, y=0, dw=10, dh=10)

output_file("image_rgba.html", title="image_rgba.py example")

show(p)  # open a browser




### PERIODIC TABLE

from bokeh.io import output_file, show
from bokeh.plotting import figure
from bokeh.sampledata.periodic_table import elements
from bokeh.transform import dodge, factor_cmap

output_file("periodic.html")

periods = ["I", "II", "III", "IV", "V", "VI", "VII"]
groups = [str(x) for x in range(1, 19)]

df = elements.copy()
df["atomic mass"] = df["atomic mass"].astype(str)
df["group"] = df["group"].astype(str)
df["period"] = [periods[x-1] for x in df.period]
df = df[df.group != "-"]
df = df[df.symbol != "Lr"]
df = df[df.symbol != "Lu"]

cmap = {
    "alkali metal"         : "#a6cee3",
    "alkaline earth metal" : "#1f78b4",
    "metal"                : "#d93b43",
    "halogen"              : "#999d9a",
    "metalloid"            : "#e08d49",
    "noble gas"            : "#eaeaea",
    "nonmetal"             : "#f1d4Af",
    "transition metal"     : "#599d7A",
}

TOOLTIPS = [
    ("Name", "@name"),
    ("Atomic number", "@{atomic number}"),
    ("Atomic mass", "@{atomic mass}"),
    ("Type", "@metal"),
    ("CPK color", "$color[hex, swatch]:CPK"),
    ("Electronic configuration", "@{electronic configuration}"),
]

p = figure(title="Periodic Table (omitting LA and AC Series)", plot_width=1000, plot_height=450,
           x_range=groups, y_range=list(reversed(periods)),
           tools="hover", toolbar_location=None, tooltips=TOOLTIPS)

r = p.rect("group", "period", 0.95, 0.95, source=df, fill_alpha=0.6, legend_field="metal",
           color=factor_cmap('metal', palette=list(cmap.values()), factors=list(cmap.keys())))

text_props = {"source": df, "text_align": "left", "text_baseline": "middle"}

x = dodge("group", -0.4, range=p.x_range)

p.text(x=x, y="period", text="symbol", text_font_style="bold", **text_props)

p.text(x=x, y=dodge("period", 0.3, range=p.y_range), text="atomic number",
       text_font_size="8pt", **text_props)

p.text(x=x, y=dodge("period", -0.35, range=p.y_range), text="name",
       text_font_size="5pt", **text_props)

p.text(x=x, y=dodge("period", -0.2, range=p.y_range), text="atomic mass",
       text_font_size="5pt", **text_props)

p.text(x=["3", "3"], y=["VI", "VII"], text=["LA", "AC"], text_align="center", text_baseline="middle")

p.outline_line_color = None
p.grid.grid_line_color = None
p.axis.axis_line_color = None
p.axis.major_tick_line_color = None
p.axis.major_label_standoff = 0
p.legend.orientation = "horizontal"
p.legend.location ="top_center"
p.hover.renderers = [r] # only hover element boxes

show(p)


from typing import Any, List, Tuple

import numpy as np

from bokeh.layouts import gridplot
from bokeh.plotting import figure, output_file, show


def streamlines(x: np.ndarray, y, u, v, density: float = 1) -> Tuple[List[Any], List[Any]]:
    ''' Return streamlines of a vector flow.

    * x and y are 1d arrays defining an *evenly spaced* grid.
    * u and v are 2d arrays (shape [y,x]) giving velocities.
    * density controls the closeness of the streamlines.

    '''

    ## Set up some constants - size of the grid used.
    NGX = len(x)
    NGY = len(y)

    ## Constants used to convert between grid index coords and user coords.
    DX = x[1]-x[0]
    DY = y[1]-y[0]
    XOFF = x[0]
    YOFF = y[0]

    ## Now rescale velocity onto axes-coordinates
    u = u / (x[-1]-x[0])
    v = v / (y[-1]-y[0])
    speed = np.sqrt(u*u+v*v)
    ## s (path length) will now be in axes-coordinates, but we must
    ## rescale u for integrations.
    u *= NGX
    v *= NGY
    ## Now u and v in grid-coordinates.

    NBX = int(30*density)
    NBY = int(30*density)

    blank = np.zeros((NBY,NBX))

    bx_spacing = NGX/float(NBX-1)
    by_spacing = NGY/float(NBY-1)

    def blank_pos(xi, yi):
        return int((xi / bx_spacing) + 0.5), \
               int((yi / by_spacing) + 0.5)

    def value_at(a, xi, yi):
        if type(xi) == np.ndarray:
            x = xi.astype(np.int)
            y = yi.astype(np.int)
        else:
            x = np.int(xi)
            y = np.int(yi)
        a00 = a[y,x]
        a01 = a[y,x+1]
        a10 = a[y+1,x]
        a11 = a[y+1,x+1]
        xt = xi - x
        yt = yi - y
        a0 = a00*(1-xt) + a01*xt
        a1 = a10*(1-xt) + a11*xt
        return a0*(1-yt) + a1*yt

    def rk4_integrate(x0, y0):
        ## This function does RK4 forward and back trajectories from
        ## the initial conditions, with the odd 'blank array'
        ## termination conditions. TODO tidy the integration loops.

        def f(xi, yi):
            dt_ds = 1./value_at(speed, xi, yi)
            ui = value_at(u, xi, yi)
            vi = value_at(v, xi, yi)
            return ui*dt_ds, vi*dt_ds

        def g(xi, yi):
            dt_ds = 1./value_at(speed, xi, yi)
            ui = value_at(u, xi, yi)
            vi = value_at(v, xi, yi)
            return -ui*dt_ds, -vi*dt_ds

        check = lambda xi, yi: xi>=0 and xi<NGX-1 and yi>=0 and yi<NGY-1

        bx_changes = []
        by_changes = []

        ## Integrator function
        def rk4(x0, y0, f):
            ds = 0.01 #min(1./NGX, 1./NGY, 0.01)
            stotal = 0
            xi = x0
            yi = y0
            xb, yb = blank_pos(xi, yi)
            xf_traj = []
            yf_traj = []
            while check(xi, yi):
                # Time step. First save the point.
                xf_traj.append(xi)
                yf_traj.append(yi)
                # Next, advance one using RK4
                try:
                    k1x, k1y = f(xi, yi)
                    k2x, k2y = f(xi + .5*ds*k1x, yi + .5*ds*k1y)
                    k3x, k3y = f(xi + .5*ds*k2x, yi + .5*ds*k2y)
                    k4x, k4y = f(xi + ds*k3x, yi + ds*k3y)
                except IndexError:
                    # Out of the domain on one of the intermediate steps
                    break
                xi += ds*(k1x+2*k2x+2*k3x+k4x) / 6.
                yi += ds*(k1y+2*k2y+2*k3y+k4y) / 6.
                # Final position might be out of the domain
                if not check(xi, yi): break
                stotal += ds
                # Next, if s gets to thres, check blank.
                new_xb, new_yb = blank_pos(xi, yi)
                if new_xb != xb or new_yb != yb:
                    # New square, so check and colour. Quit if required.
                    if blank[new_yb,new_xb] == 0:
                        blank[new_yb,new_xb] = 1
                        bx_changes.append(new_xb)
                        by_changes.append(new_yb)
                        xb = new_xb
                        yb = new_yb
                    else:
                        break
                if stotal > 2:
                    break
            return stotal, xf_traj, yf_traj

        integrator = rk4

        sf, xf_traj, yf_traj = integrator(x0, y0, f)
        sb, xb_traj, yb_traj = integrator(x0, y0, g)
        stotal = sf + sb
        x_traj = xb_traj[::-1] + xf_traj[1:]
        y_traj = yb_traj[::-1] + yf_traj[1:]

        ## Tests to check length of traj. Remember, s in units of axes.
        if len(x_traj) < 1: return None
        if stotal > .2:
            initxb, inityb = blank_pos(x0, y0)
            blank[inityb, initxb] = 1
            return x_traj, y_traj
        else:
            for xb, yb in zip(bx_changes, by_changes):
                blank[yb, xb] = 0
            return None

    ## A quick function for integrating trajectories if blank==0.
    trajectories = []
    def traj(xb, yb):
        if xb < 0 or xb >= NBX or yb < 0 or yb >= NBY:
            return
        if blank[yb, xb] == 0:
            t = rk4_integrate(xb*bx_spacing, yb*by_spacing)
            if t is not None:
                trajectories.append(t)

    ## Now we build up the trajectory set. I've found it best to look
    ## for blank==0 along the edges first, and work inwards.
    for indent in range((max(NBX,NBY))//2):
        for xi in range(max(NBX,NBY)-2*indent):
            traj(xi+indent, indent)
            traj(xi+indent, NBY-1-indent)
            traj(indent, xi+indent)
            traj(NBX-1-indent, xi+indent)

    xs = [np.array(t[0])*DX+XOFF for t in trajectories]
    ys = [np.array(t[1])*DY+YOFF for t in trajectories]

    return xs, ys

xx = np.linspace(-3, 3, 100)
yy = np.linspace(-3, 3, 100)

Y, X = np.meshgrid(xx, yy)
U = -1 - X**2 + Y
V = 1 + X - Y**2
speed = np.sqrt(U*U + V*V)
theta = np.arctan(V/U)

x0 = X[::2, ::2].flatten()
y0 = Y[::2, ::2].flatten()
length = speed[::2, ::2].flatten()/40
angle = theta[::2, ::2].flatten()
x1 = x0 + length * np.cos(angle)
y1 = y0 + length * np.sin(angle)

xs, ys = streamlines(xx, yy, U.T, V.T, density=2)

cm = np.array(["#C7E9B4", "#7FCDBB", "#41B6C4", "#1D91C0", "#225EA8", "#0C2C84"])
ix = ((length-length.min())/(length.max()-length.min())*5).astype('int')
colors = cm[ix]

p1 = figure(x_range=(-3,3 ), y_range=(-3, 3))
p1.segment(x0, y0, x1, y1, color=colors, line_width=2)

p2 = figure(x_range=p1.x_range, y_range=p1.y_range)
p2.multi_line(xs, ys, color="#ee6666", line_width=2, line_alpha=0.8)

output_file("vector.html", title="vector.py example")

show(gridplot([[p1,p2]], plot_width=400, plot_height=400))  # open a browser


from typing import Any, List, Tuple

import numpy as np

from bokeh.layouts import gridplot
from bokeh.plotting import figure, output_file, show


def streamlines(x: np.ndarray, y, u, v, density: float = 1) -> Tuple[List[Any], List[Any]]:
    ''' Return streamlines of a vector flow.

    * x and y are 1d arrays defining an *evenly spaced* grid.
    * u and v are 2d arrays (shape [y,x]) giving velocities.
    * density controls the closeness of the streamlines.

    '''

    ## Set up some constants - size of the grid used.
    NGX = len(x)
    NGY = len(y)

    ## Constants used to convert between grid index coords and user coords.
    DX = x[1]-x[0]
    DY = y[1]-y[0]
    XOFF = x[0]
    YOFF = y[0]

    ## Now rescale velocity onto axes-coordinates
    u = u / (x[-1]-x[0])
    v = v / (y[-1]-y[0])
    speed = np.sqrt(u*u+v*v)
    ## s (path length) will now be in axes-coordinates, but we must
    ## rescale u for integrations.
    u *= NGX
    v *= NGY
    ## Now u and v in grid-coordinates.

    NBX = int(30*density)
    NBY = int(30*density)

    blank = np.zeros((NBY,NBX))

    bx_spacing = NGX/float(NBX-1)
    by_spacing = NGY/float(NBY-1)

    def blank_pos(xi, yi):
        return int((xi / bx_spacing) + 0.5), \
               int((yi / by_spacing) + 0.5)

    def value_at(a, xi, yi):
        if type(xi) == np.ndarray:
            x = xi.astype(np.int)
            y = yi.astype(np.int)
        else:
            x = np.int(xi)
            y = np.int(yi)
        a00 = a[y,x]
        a01 = a[y,x+1]
        a10 = a[y+1,x]
        a11 = a[y+1,x+1]
        xt = xi - x
        yt = yi - y
        a0 = a00*(1-xt) + a01*xt
        a1 = a10*(1-xt) + a11*xt
        return a0*(1-yt) + a1*yt

    def rk4_integrate(x0, y0):
        ## This function does RK4 forward and back trajectories from
        ## the initial conditions, with the odd 'blank array'
        ## termination conditions. TODO tidy the integration loops.

        def f(xi, yi):
            dt_ds = 1./value_at(speed, xi, yi)
            ui = value_at(u, xi, yi)
            vi = value_at(v, xi, yi)
            return ui*dt_ds, vi*dt_ds

        def g(xi, yi):
            dt_ds = 1./value_at(speed, xi, yi)
            ui = value_at(u, xi, yi)
            vi = value_at(v, xi, yi)
            return -ui*dt_ds, -vi*dt_ds

        check = lambda xi, yi: xi>=0 and xi<NGX-1 and yi>=0 and yi<NGY-1

        bx_changes = []
        by_changes = []

        ## Integrator function
        def rk4(x0, y0, f):
            ds = 0.01 #min(1./NGX, 1./NGY, 0.01)
            stotal = 0
            xi = x0
            yi = y0
            xb, yb = blank_pos(xi, yi)
            xf_traj = []
            yf_traj = []
            while check(xi, yi):
                # Time step. First save the point.
                xf_traj.append(xi)
                yf_traj.append(yi)
                # Next, advance one using RK4
                try:
                    k1x, k1y = f(xi, yi)
                    k2x, k2y = f(xi + .5*ds*k1x, yi + .5*ds*k1y)
                    k3x, k3y = f(xi + .5*ds*k2x, yi + .5*ds*k2y)
                    k4x, k4y = f(xi + ds*k3x, yi + ds*k3y)
                except IndexError:
                    # Out of the domain on one of the intermediate steps
                    break
                xi += ds*(k1x+2*k2x+2*k3x+k4x) / 6.
                yi += ds*(k1y+2*k2y+2*k3y+k4y) / 6.
                # Final position might be out of the domain
                if not check(xi, yi): break
                stotal += ds
                # Next, if s gets to thres, check blank.
                new_xb, new_yb = blank_pos(xi, yi)
                if new_xb != xb or new_yb != yb:
                    # New square, so check and colour. Quit if required.
                    if blank[new_yb,new_xb] == 0:
                        blank[new_yb,new_xb] = 1
                        bx_changes.append(new_xb)
                        by_changes.append(new_yb)
                        xb = new_xb
                        yb = new_yb
                    else:
                        break
                if stotal > 2:
                    break
            return stotal, xf_traj, yf_traj

        integrator = rk4

        sf, xf_traj, yf_traj = integrator(x0, y0, f)
        sb, xb_traj, yb_traj = integrator(x0, y0, g)
        stotal = sf + sb
        x_traj = xb_traj[::-1] + xf_traj[1:]
        y_traj = yb_traj[::-1] + yf_traj[1:]

        ## Tests to check length of traj. Remember, s in units of axes.
        if len(x_traj) < 1: return None
        if stotal > .2:
            initxb, inityb = blank_pos(x0, y0)
            blank[inityb, initxb] = 1
            return x_traj, y_traj
        else:
            for xb, yb in zip(bx_changes, by_changes):
                blank[yb, xb] = 0
            return None

    ## A quick function for integrating trajectories if blank==0.
    trajectories = []
    def traj(xb, yb):
        if xb < 0 or xb >= NBX or yb < 0 or yb >= NBY:
            return
        if blank[yb, xb] == 0:
            t = rk4_integrate(xb*bx_spacing, yb*by_spacing)
            if t is not None:
                trajectories.append(t)

    ## Now we build up the trajectory set. I've found it best to look
    ## for blank==0 along the edges first, and work inwards.
    for indent in range((max(NBX,NBY))//2):
        for xi in range(max(NBX,NBY)-2*indent):
            traj(xi+indent, indent)
            traj(xi+indent, NBY-1-indent)
            traj(indent, xi+indent)
            traj(NBX-1-indent, xi+indent)

    xs = [np.array(t[0])*DX+XOFF for t in trajectories]
    ys = [np.array(t[1])*DY+YOFF for t in trajectories]

    return xs, ys

xx = np.linspace(-3, 3, 100)
yy = np.linspace(-3, 3, 100)

Y, X = np.meshgrid(xx, yy)
U = -1 - X**2 + Y
V = 1 + X - Y**2
speed = np.sqrt(U*U + V*V)
theta = np.arctan(V/U)

x0 = X[::2, ::2].flatten()
y0 = Y[::2, ::2].flatten()
length = speed[::2, ::2].flatten()/40
angle = theta[::2, ::2].flatten()
x1 = x0 + length * np.cos(angle)
y1 = y0 + length * np.sin(angle)

xs, ys = streamlines(xx, yy, U.T, V.T, density=2)

cm = np.array(["#C7E9B4", "#7FCDBB", "#41B6C4", "#1D91C0", "#225EA8", "#0C2C84"])
ix = ((length-length.min())/(length.max()-length.min())*5).astype('int')
colors = cm[ix]

p1 = figure(x_range=(-3,3 ), y_range=(-3, 3))
p1.segment(x0, y0, x1, y1, color=colors, line_width=2)

p2 = figure(x_range=p1.x_range, y_range=p1.y_range)
p2.multi_line(xs, ys, color="#ee6666", line_width=2, line_alpha=0.8)

output_file("vector.html", title="vector.py example")

show(gridplot([[p1,p2]], plot_width=400, plot_height=400))  # open a browser


import numpy as np

from bokeh.plotting import figure, output_file, show
from bokeh.sampledata.les_mis import data

nodes = data['nodes']
names = [node['name'] for node in sorted(data['nodes'], key=lambda x: x['group'])]

N = len(nodes)
counts = np.zeros((N, N))
for link in data['links']:
    counts[link['source'], link['target']] = link['value']
    counts[link['target'], link['source']] = link['value']

colormap = ["#444444", "#a6cee3", "#1f78b4", "#b2df8a", "#33a02c", "#fb9a99",
            "#e31a1c", "#fdbf6f", "#ff7f00", "#cab2d6", "#6a3d9a"]

xname = []
yname = []
color = []
alpha = []
for i, node1 in enumerate(nodes):
    for j, node2 in enumerate(nodes):
        xname.append(node1['name'])
        yname.append(node2['name'])

        alpha.append(min(counts[i,j]/4.0, 0.9) + 0.1)

        if node1['group'] == node2['group']:
            color.append(colormap[node1['group']])
        else:
            color.append('lightgrey')

data=dict(
    xname=xname,
    yname=yname,
    colors=color,
    alphas=alpha,
    count=counts.flatten(),
)

p = figure(title="Les Mis Occurrences",
           x_axis_location="above", tools="hover,save",
           x_range=list(reversed(names)), y_range=names,
           tooltips = [('names', '@yname, @xname'), ('count', '@count')])

p.plot_width = 800
p.plot_height = 800
p.grid.grid_line_color = None
p.axis.axis_line_color = None
p.axis.major_tick_line_color = None
p.axis.major_label_text_font_size = "5pt"
p.axis.major_label_standoff = 0
p.xaxis.major_label_orientation = np.pi/3

p.rect('xname', 'yname', 0.9, 0.9, source=data,
       color='colors', alpha='alphas', line_color=None,
       hover_line_color='black', hover_color='colors')

output_file("les_mis.html", title="les_mis.py example")

show(p) # show the plot