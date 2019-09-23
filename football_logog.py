#!/usr/bin/env python
# coding: utf-8

# # Data Visualization
# * we will need to install numpy and matplotlib if they are not already installed
#   * we can do this using Anaconda's package manager:
#     * __`~/anaconda3/bin/conda install numpy`__
#     * __`~/anaconda3/bin/conda install matplotlib`__
# * now let's tell Jupyter we want visualizations from matplotlib to be displayed inline...

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[40]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg 
import cv2
from skimage.measure import compare_ssim
from os import listdir
from os.path import isfile, join

# x = np.linspace(0, 10, 100)
# fig = plt.figure()
# plt.plot(x, np.sin(x), 'r:')
# plt.plot(x, np.cos(x), '-.');
# the semi-colon causes the function's return value to be discarded
# ...try it without the semi-colon

# Read Images 
# titans_img = mpimg.imread('/Users/dzhuo/Downloads/football-logo/titans.png')
# seahawks_img = mpimg.imread('/Users/dzhuo/Downloads/football-logo/seahawks.png')
path = '/Users/dzhuo/Downloads/football-logo/'
files = [f for f in listdir(path) if isfile(join(path, f))]
D = {}
for f in files:
    name = f.split('.')[0]
    image = cv2.imread(join(path, f))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (300, 200), interpolation = cv2.INTER_AREA)
    D[name] = image

target_image_file = '/Users/dzhuo/Downloads/football-logo/patriots.png'
target_img = cv2.imread(target_image_file, cv2.IMREAD_UNCHANGED)
modified_target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
modified_target_img = cv2.resize(modified_target_img, (300, 200), interpolation = cv2.INTER_AREA)

best_score = -1
team_name = 'N/A'
for name, img in D.items():
    score, _ = compare_ssim(img, modified_target_img, full=True)
    if score > best_score:
        best_score = score
        team_name = name

print("Team Name: {}".format(team_name))
plt.imshow(mpimg.imread(target_image_file))


# ## You can save your plots...

# In[5]:


fig.savefig('my_figure.png')


# In[6]:


get_ipython().system('ls -lh my_figure.png')
# On Windows, comment out above and uncomment below
#!dir my_figure.png


# ## ...and reload saved images for display inside the notebook

# In[7]:


from IPython.display import Image
Image('my_figure.png')


# In[8]:


# matplotlib supports many different file types
fig.canvas.get_supported_filetypes()


# ## MATLAB-Style Interface

# In[10]:


plt.figure()  # create a plot figure

# create the first of two panels and set current axis
plt.subplot(2, 1, 1) # (rows, columns, panel number)
plt.plot(x, np.sin(x))

# create the second panel and set current axis
plt.subplot(2, 1, 2)
plt.plot(x, np.cos(x));


# ## Object-Oriented Interface

# In[11]:


# First create a grid of plots
# ax will be an array of two Axes objects
fig, ax = plt.subplots(2)

# Call plot() method on the appropriate object
ax[0].plot(x, np.sin(x))
ax[1].plot(x, np.cos(x));


# ## Grids

# In[12]:


plt.style.use('seaborn-whitegrid')
fig = plt.figure()
ax = plt.axes()


# ## Draw a Function

# In[13]:


plt.style.use('seaborn-whitegrid')
fig = plt.figure()
ax = plt.axes()
x = np.linspace(0, 10, 1000)
ax.plot(x, np.sin(x));


# ## Many ways to specify color...

# In[14]:


plt.plot(x, np.sin(x - 0), color='blue')        # specify color by name
plt.plot(x, np.sin(x - 1), color='g')           # short color code (rgbcmyk)
plt.plot(x, np.sin(x - 2), color='0.75')        # Grayscale between 0 and 1
plt.plot(x, np.sin(x - 3), color='#FFDD44')     # Hex code (RRGGBB from 00 to FF)
plt.plot(x, np.sin(x - 4), color=(1.0,0.2,0.3)) # RGB tuple, values 0 to 1
plt.plot(x, np.sin(x - 5), color='chartreuse'); # all HTML color names supported


# ## Specifying different line styles...

# In[15]:


plt.plot(x, x + 0, linestyle='solid')
plt.plot(x, x + 1, linestyle='dashed')
plt.plot(x, x + 2, linestyle='dashdot')
plt.plot(x, x + 3, linestyle='dotted');

# For short, you can use the following codes:
plt.plot(x, x + 4, linestyle='-')  # solid
plt.plot(x, x + 5, linestyle='--') # dashed
plt.plot(x, x + 6, linestyle='-.') # dashdot
plt.plot(x, x + 7, linestyle=':');  # dotted


# ## Specify axes limits...

# In[16]:


plt.plot(x, np.sin(x))

plt.xlim(-1, 11)
plt.ylim(-1.5, 1.5);


# ## Flipping the Axes Limits

# In[17]:


plt.plot(x, np.sin(x))

plt.xlim(10, 0)
plt.ylim(1.2, -1.2);


# ## Axis

# In[18]:


plt.plot(x, np.sin(x))
plt.axis([-1, 11, -1.5, 1.5]);


# ## ...or let matplotlib "tighten" the axes...

# In[8]:


plt.plot(x, np.sin(x))
plt.axis('tight');


# ## ...or make the limits equal

# In[9]:


plt.plot(x, np.sin(x))
plt.axis('equal');


# ## Add titles and axis labels

# In[21]:


plt.plot(x, np.sin(x))
plt.title("A Sine Curve")
plt.xlabel("x")
plt.ylabel("sin(x)");


# # ...and a legend

# In[11]:


plt.plot(x, np.sin(x), '-g', label='sin(x)')
plt.plot(x, np.cos(x), ':b', label='cos(x)')
plt.axis('equal')

plt.legend();


# ## OO interface to axes

# In[14]:


ax = plt.axes()
ax.plot(x, np.sin(x))
# ax.set(xlim=(0, 10), ylim=(-2, 2),
#        xlabel='x', ylabel='sin(x)',
#        title='A Simple Plot');


# ## Interface Differences
# | MATLAB-Style | OO Style        |
# |--------------|-----------------|
# | plt.xlabel() | ax.set_xlabel() |
# | plt.ylabel() | ax.set_ylabel() |
# | plt.xlim()   | ax.set_xlim()   |
# | plt.ylim()   | ax.set_ylim()   |
# | plt.title()  | ax.set_title()  |

# ## Specify different plot markers 

# In[20]:


rng = np.random.RandomState(0)

for marker in 'o.,x+v^<>sd':
    plt.plot(rng.rand(3), rng.rand(3), marker,
             label='marker={}'.format(marker))
plt.legend(numpoints=1)
plt.xlim(0, 1.8);


# ## Scatterplots with Colors and Sizes

# In[24]:


rng = np.random.RandomState(0)
x = rng.randn(100)
y = rng.randn(100)
colors = rng.rand(100)
sizes = 1000 * rng.rand(100)

plt.scatter(x, y, c=colors, s=sizes, alpha=0.3,
            cmap='viridis')
plt.colorbar();  # show color scale


# ## Visualizing Multiple Dimensions

# In[31]:


from sklearn.datasets import load_iris
iris = load_iris()
iris.data[1]
iris.feature_names

features = iris.data.T
plt.scatter(features[0], features[1], alpha=0.2,
            s=100*features[3], c=iris.target, cmap='viridis')
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1]);


# ## Histograms

# In[35]:


data = np.random.randn(1_000_000)

plt.hist(data);


# In[38]:


plt.hist(data, bins=50, density=True, alpha=0.5,
         histtype='stepfilled', color='steelblue',
         edgecolor='none');


# ## Custom legends

# In[29]:


plt.style.use('classic')
x = np.linspace(0, 10, 1000)
fig, ax = plt.subplots()
ax.plot(x, np.sin(x), '-b', label='Sine')
ax.plot(x, np.cos(x), '--r', label='Cosine')
ax.axis('equal')
leg = ax.legend()


# In[30]:


ax.legend(loc='upper left', frameon=False)
fig


# In[31]:


ax.legend(frameon=False, loc='lower center', ncol=2)
fig


# ## Display a grid of images

# In[32]:


# load images of the digits 0 through 5 and visualize several of them
from sklearn.datasets import load_digits
digits = load_digits(n_class=6)

fig, ax = plt.subplots(8, 8, figsize=(6, 6))
for i, axi in enumerate(ax.flat):
    axi.imshow(digits.images[i], cmap='binary')
    axi.set(xticks=[], yticks=[])


# ## Calculate birth data using pandas

# In[33]:


import matplotlib.pyplot as plt
import matplotlib as mpl
plt.style.use('seaborn-whitegrid')
import numpy as np
import pandas as pd

births = pd.read_csv('data/births.csv')

# From Python Data Science Handbook by Jake VanderPlas
# clean the data using a "robust sigma-clipping operation"
quartiles = np.percentile(births['births'], [25, 50, 75])
mu = quartiles[1]
# a robust estimate of the std dev
# 0.74 is from the IQR of a Gaussian distribution
sig = 0.74 * (quartiles[2] - quartiles[0])
births = births.query('(births > @mu - 5 * @sig) & (births < @mu + 5 * @sig)')

births['day'] = births['day'].astype(int)

births.index = pd.to_datetime(10000 * births.year +
                              100 * births.month +
                              births.day, format='%Y%m%d')
births_by_date = births.pivot_table('births',
                                    [births.index.month, births.index.day])
births_by_date.index = [pd.datetime(2012, month, day)
                        for (month, day) in births_by_date.index]


# ## ...and display

# In[34]:


fig, ax = plt.subplots(figsize=(12, 4))
births_by_date.plot(ax=ax);


# ## ...but that plot would look much better with annotations

# In[35]:


fig, ax = plt.subplots(figsize=(12, 4))
births_by_date.plot(ax=ax)

# Add labels to the plot
ax.annotate("New Year's Day", xy=('2012-1-1', 4100),  xycoords='data',
            xytext=(50, -30), textcoords='offset points',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3,rad=-0.2"))

ax.annotate("Independence Day", xy=('2012-7-4', 4250),  xycoords='data',
            bbox=dict(boxstyle="round", fc="none", ec="gray"),
            xytext=(10, -40), textcoords='offset points', ha='center',
            arrowprops=dict(arrowstyle="->"))

ax.annotate('Labor Day', xy=('2012-9-4', 4850), xycoords='data', ha='center',
            xytext=(0, -20), textcoords='offset points')
ax.annotate('', xy=('2012-9-1', 4850), xytext=('2012-9-7', 4850),
            xycoords='data', textcoords='data',
            arrowprops={'arrowstyle': '|-|,widthA=0.2,widthB=0.2', })

ax.annotate('Halloween', xy=('2012-10-31', 4600),  xycoords='data',
            xytext=(-80, -40), textcoords='offset points',
            arrowprops=dict(arrowstyle="fancy",
                            fc="0.6", ec="none",
                            connectionstyle="angle3,angleA=0,angleB=-90"))

ax.annotate('Thanksgiving', xy=('2012-11-25', 4500),  xycoords='data',
            xytext=(-120, -60), textcoords='offset points',
            bbox=dict(boxstyle="round4,pad=.5", fc="0.9"),
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="angle,angleA=0,angleB=80,rad=20"))


ax.annotate('Christmas', xy=('2012-12-25', 3850),  xycoords='data',
             xytext=(-30, 0), textcoords='offset points',
             size=13, ha='right', va="center",
             bbox=dict(boxstyle="round", alpha=0.1),
             arrowprops=dict(arrowstyle="wedge,tail_width=0.5", alpha=0.1));

# Label the axes
ax.set(title='USA births by day of year (1969-1988)',
       ylabel='average daily births')

# Format the x axis with centered month labels
ax.xaxis.set_major_locator(mpl.dates.MonthLocator())
ax.xaxis.set_minor_locator(mpl.dates.MonthLocator(bymonthday=15))
ax.xaxis.set_major_formatter(plt.NullFormatter())
ax.xaxis.set_minor_formatter(mpl.dates.DateFormatter('%h'));

ax.set_ylim(3600, 5400);


# In[ ]:




