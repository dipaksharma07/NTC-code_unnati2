#!/usr/bin/env python
# coding: utf-8

# # UNIT - 3 Data Visualization

# In[1]:


#import pylot from matplotlib

import matplotlib.pyplot as plt


# In[6]:


x = [1,2,3]

y = [4,5,6]

plt.plot(x,y)

plt.show


# In[26]:


import numpy as np

x = np.array([2,3,8,7])
y = np.array([1,4,1,9])

plt.plot(x,y, marker ='o', c = 'b', linestyle = 'dotted')

plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.title("Graph")
plt.show


# In[2]:


#import matplotlib,pyplot as plt
import matplotlib.pyplot as plt
import random as ran

students = ["deepak","chirag","jay","arjun","lucky"]

marks=[]
for i in range(0,len(students)):
    marks.append(ran.randint(0,101))
    
plt.xlabel("students")
plt.xticks(rotation = "30")
plt.ylabel("Marks")
plt.title("CLASS RECORDS")
plt.plot(students,marks, 'm--',
        marker = 'o',color = 'b',markerfacecolor = 'r',markersize = '20')


# In[4]:


# importing the required module
import matplotlib.pyplot as plt
# x axis values
x = [1,2,3]
# corresponding y axis values 
y = [2,4,1]
# ploting the points
plt.plot(x,y)
# naming the x axis 
plt.xlabel('x - axis')
# naming the y axis
plt.ylabel('y -axis')
# giving title to my graph
plt.title('My first Graph')


# In[15]:


import numpy as np
x = np.arange(0,10,0.1)

y = np.sin(x)


#figure in matplotlib using figure()
plt.figure(figsize=(10,8))
plt.plot(x,y)
plt.show()


# # Sub plot

# In[36]:


#plot1:
x = np.array([0,1,2,3])
y = np.array([3,8,1,10])

#plt.subplot(raws, colums, plot no.)
plt.subplot(1,3,1)
plt.plot(x,y)

#plot2:
x = np.array([0,1,2,3])
y = np.array([10,20,30,40])

plt.subplot(3,3,2)
plt.plot(x,y)

#plot:3
x = np.array([0,1,2,3])
y = np.array([10,20,30,40])

plt.subplot(1,3,3)
plt.plot(x,y)

plt.suptitle("subplotting")


# In[35]:


ypoint = np.array([3,8,1,10])
plt.plot(ypoint, marker= 'o', ms='19', color ='r')


# # Bar plot

# In[40]:


#importing matplotlib module
from matplotlib import pyplot as plt

# x-axis values
x = [5,2,9,4,7]
# y-axis values
y = [10,5,8,4,2]

plt.bar(x,y)

plt.show()


# In[41]:


#scatter plot 
# importing matplotlib module
from matplotlib import pyplot as plt

# x-axis values
x = [5,2,9,4,7,3,8,20,1]
# y-axis values
y = [10,5,8,4,2,1,6,3,9]

plt.scatter(x,y)

plt.show()


# In[50]:


# Ticks are the marker denoting daata points on axes.
# importing libreries

# values of x and y axes
x = [5,10,15,20,25,30,35,40,45,50]
y = [1,4,3,2,7,6,9,8,10,5]

plt.plot(x,y,'g')
plt.xlabel('X')
plt.ylabel('Y')

# here we set the size for ticks, rotation and color value
plt.tick_params(axis='x',labelsize=18, labelrotation=90, labelcolor='r')
plt.tick_params(axis='y',labelsize=12, labelrotation=20, labelcolor='b')

plt.show()


# In[54]:


#adding legend in a graph
# importing module 
# Y-axis values
y1 = [2,3,4.5]

# Y-axis values
y2 = [1,1.5,5]

# function to show the plot 
plt.plot(y1)
plt.plot(y2)

#FUNCTION ADD LEGEND
plt.legend(['blue','orange'], loc = 'upper left')

plt.show()


# # Histogram

# In[56]:


#Histogram

data = np.random.normal(170,10,250)
print(data)

#the hist() function will read the array and produce a histogram 

#set the bin value 

plt.hist(data);

plt.show()
plt.hist(data, bins=30);
plt.show()


# In[63]:


# draw plot a figure 

fig, ax = plt.subplots()

x = np.linspace(0, 20, 1000)
ax.plot(x, np.cos(x))
ax.axis('equal')

#text annotation to the plot where it indicate maximum value of the curv

ax.annotate('local maximum', xy=(6.28,1), xytext=(10,4),
           arrowprops=dict(facecolor='black', shrink=0.04))
#text annotation to the plot where it indicate minimum value of the curv
ax.annotate('local minimum', xy=(5* np.pi, -1), xytext=(2, -6),
           arrowprops=dict(arrowstyle="->",
                           connectionstyle="angle3,angleA=0,angleB=-90"));


# # Adding Annotate & legend

# In[24]:


import numpy as np
import matplotlib.pyplot as plt
#you can also write 
#from matplotlib import pyplot as plt


x = np.arange(0,10)
y = x^2
z = x^3

plt.plot(x,y)
plt.plot(y,z)
plt.plot(x,z)

plt.title("Graph")
plt.xlabel('Time')
plt.ylabel('Distance')


#Annotation

plt.annotate(xy=[2,1],text="First Entery", rotation= 15)
plt.annotate(xy=[6,4],text="2nd Entery", rotation= 15)
plt.annotate(xy=[8,10],text="3rd Entery", rotation= 15)

#Legend
plt.legend(['Race 1','Race 2','Race 3'], loc= 2)

plt.show()


# # Visualization using numpy

# In[25]:


x = np.arange(0,3*np.pi,0.1)
y = np.tan(x)

plt.plot(x,y)
plt.show()


# # Pie plot 
# 
# pie plot is circular satastical plot that can dispaly only one series of data.
# the area of chart is that total % of the given data.

# In[31]:


#import libreries

#creating dotset

Branch = ["computer","Electrical","Mechanical","civil","automobile","IT"]

data = [50,34,40,29,12,23]
explode = (0.15,0,0,0.1,0,0)
#create chart 
fig = plt.figure(figsize=(10,7))
plt.pie(data, labels= Branch, explode=explode, autopct= '%1.1f%%', shadow = True)

plt.show()


# # 3D ploting

# In[33]:


#import mplot3d toolkit, numpy & matplotlib

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d #3d toolkit lib


# In[36]:


fig = plt.figure()

#syntex for 3D projection 

ax = plt.axes(projection = '3d')

#defining all 3 axes 

z = np.linspace(0,1,100)
x = z*np.sin(25*z)
y = z*np.cos(25*z)

ax.plot3D(x, y, z, 'green')
ax.set_title('3D ploting')
plt.show()


# # Saving the chart

# In[41]:


#import matplotlib,pyplot as plt
import matplotlib.pyplot as plt
import random as ran

students = ["deepak","chirag","jay","arjun","lucky"]

marks=[]
for i in range(0,len(students)):
    marks.append(ran.randint(0,101))
    
plt.xlabel("students")
plt.xticks(rotation = "30")
plt.ylabel("Marks")
plt.title("CLASS RECORDS")
plt.plot(students,marks, 'm--',
        marker = 'o',color = 'b',markerfacecolor = 'r',markersize = '20')

plt.savefig("class.pdf", formet = 'pdf', dpi=200)


# In[ ]:




