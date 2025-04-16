"""
COMPREHENSIVE MATPLOTLIB REFERENCE
Covering all major plotting types and customization options
"""

import matplotlib.pyplot as plt
import numpy as np

# ==============================================
# 1. BASIC PLOTTING
# ==============================================
print("\n=== 1. Basic Plotting ===\n")

# Create sample data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Line plot
plt.figure(figsize=(8, 4))
plt.plot(x, y, label='sin(x)', color='blue', linestyle='-', linewidth=2)
plt.title("Basic Line Plot")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.legend()
plt.grid(True)
plt.show()

# Scatter plot
plt.figure(figsize=(8, 4))
plt.scatter(x, y, c='red', marker='o', s=20, alpha=0.7, label='Points')
plt.title("Scatter Plot")
plt.xlabel("X values")
plt.ylabel("Y values")
plt.legend()
plt.show()

# ==============================================
# 2. MULTIPLE PLOTS & SUBPLOTS
# ==============================================
print("\n=== 2. Multiple Plots & Subplots ===\n")

# Multiple lines on same plot
plt.figure(figsize=(8, 4))
plt.plot(x, np.sin(x), label='sin(x)')
plt.plot(x, np.cos(x), label='cos(x)')
plt.title("Multiple Lines")
plt.legend()
plt.show()

# Subplots
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes[0, 0].plot(x, np.sin(x))
axes[0, 0].set_title("sin(x)")
axes[0, 1].scatter(x, np.cos(x), c='green')
axes[0, 1].set_title("cos(x) scatter")
axes[1, 0].bar([1, 2, 3], [3, 7, 2])
axes[1, 0].set_title("Bar chart")
axes[1, 1].hist(np.random.randn(1000), bins=30)
axes[1, 1].set_title("Histogram")
plt.tight_layout()
plt.show()

# ==============================================
# 3. SPECIALIZED PLOTS
# ==============================================
print("\n=== 3. Specialized Plots ===\n")

# Bar plots
categories = ['A', 'B', 'C', 'D']
values = [15, 25, 30, 20]

plt.figure(figsize=(8, 4))
plt.bar(categories, values, color=['red', 'green', 'blue', 'cyan'])
plt.title("Bar Chart")
plt.xlabel("Categories")
plt.ylabel("Values")
plt.show()

# Horizontal bar plot
plt.figure(figsize=(8, 4))
plt.barh(categories, values, color='purple')
plt.title("Horizontal Bar Chart")
plt.show()

# Stacked bar plot
men_means = [20, 35, 30, 35]
women_means = [25, 32, 34, 20]

plt.figure(figsize=(8, 4))
plt.bar(categories, men_means, label='Men')
plt.bar(categories, women_means, bottom=men_means, label='Women')
plt.title("Stacked Bar Chart")
plt.legend()
plt.show()

# Histograms
data = np.random.randn(1000)

plt.figure(figsize=(8, 4))
plt.hist(data, bins=30, density=True, alpha=0.7, 
         histtype='stepfilled', color='steelblue', edgecolor='black')
plt.title("Histogram")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()

# Pie chart
sizes = [15, 30, 45, 10]
labels = ['A', 'B', 'C', 'D']
explode = (0, 0.1, 0, 0)  # "Explode" the 2nd slice

plt.figure(figsize=(6, 6))
plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
plt.title("Pie Chart")
plt.show()

# Box plot
data = [np.random.normal(0, std, 100) for std in range(1, 4)]

plt.figure(figsize=(8, 4))
plt.boxplot(data, vert=True, patch_artist=True)
plt.title("Box Plot")
plt.xticks([1, 2, 3], ['A', 'B', 'C'])
plt.show()

# Violin plot
plt.figure(figsize=(8, 4))
plt.violinplot(data, showmeans=True, showmedians=True)
plt.title("Violin Plot")
plt.xticks([1, 2, 3], ['A', 'B', 'C'])
plt.show()

# ==============================================
# 4. ADVANCED PLOTTING
# ==============================================
print("\n=== 4. Advanced Plotting ===\n")

# Contour plot
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = np.sin(np.sqrt(X**2 + Y**2))

plt.figure(figsize=(8, 6))
contour = plt.contour(X, Y, Z, levels=20, cmap='RdGy')
plt.colorbar(contour)
plt.title("Contour Plot")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# 3D Surface plot
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
fig.colorbar(surf)
ax.set_title("3D Surface Plot")
plt.show()

# Polar plot
r = np.arange(0, 2, 0.01)
theta = 2 * np.pi * r

plt.figure(figsize=(6, 6))
ax = plt.subplot(111, polar=True)
ax.plot(theta, r)
ax.set_title("Polar Plot", va='bottom')
plt.show()

# Quiver plot (vector field)
x, y = np.meshgrid(np.arange(-2, 2, 0.2), np.arange(-2, 2, 0.2))
u = -y
v = x

plt.figure(figsize=(8, 6))
plt.quiver(x, y, u, v, scale=50)
plt.title("Quiver Plot (Vector Field)")
plt.show()

# ==============================================
# 5. CUSTOMIZATION & STYLING
# ==============================================
print("\n=== 5. Customization & Styling ===\n")

# Customizing plots
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

plt.figure(figsize=(10, 5))
plt.plot(x, y1, label='sin(x)', color='red', linestyle='--', linewidth=2)
plt.plot(x, y2, label='cos(x)', color='blue', linestyle=':', linewidth=3)

# Customize axes
plt.xlim(0, 10)
plt.ylim(-1.5, 1.5)
plt.xticks(np.arange(0, 11, 2))
plt.yticks(np.arange(-1, 1.1, 0.5))

# Add text and annotations
plt.title("Customized Plot", fontsize=16, fontweight='bold')
plt.xlabel("X-axis", fontsize=14)
plt.ylabel("Y-axis", fontsize=14)
plt.text(5, 0, "Intersection Point", fontsize=12, ha='center')
plt.annotate('Max value', xy=(np.pi/2, 1), xytext=(3, 1.2),
             arrowprops=dict(facecolor='black', shrink=0.05))

# Grid and legend
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='upper right', fontsize=12)

# Custom spines
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('gray')
ax.spines['bottom'].set_color('gray')

plt.show()

# Styles
print("Available styles:", plt.style.available)
plt.style.use('ggplot')

plt.figure(figsize=(8, 4))
plt.plot(x, y1, label='sin(x)')
plt.plot(x, y2, label='cos(x)')
plt.title("Plot with ggplot Style")
plt.legend()
plt.show()

plt.style.use('default')  # Reset to default

# ==============================================
# 6. SAVING & EXPORTING
# ==============================================
print("\n=== 6. Saving & Exporting ===\n")

plt.figure(figsize=(8, 4))
plt.plot(x, y)
plt.title("Plot to be Saved")

# Save in different formats
plt.savefig('plot.png', dpi=300, bbox_inches='tight')  # PNG
plt.savefig('plot.jpg', quality=90)  # JPEG
plt.savefig('plot.pdf')  # PDF
plt.savefig('plot.svg')  # SVG

print("Plot saved in multiple formats")

# ==============================================
# 7. INTERACTIVE FEATURES
# ==============================================
print("\n=== 7. Interactive Features ===\n")

# Simple interactive plot
plt.figure(figsize=(8, 4))
plt.plot(x, y)
plt.title("Interactive Plot - Try Zoom/Pan")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.show()  # Try interactive features in your environment

# Widgets (requires ipywidgets in Jupyter)
# from ipywidgets import interact
# 
# def plot_func(freq=1):
#     plt.figure(figsize=(8, 4))
#     plt.plot(x, np.sin(freq * x))
#     plt.title(f"Sine Wave with Frequency {freq}")
#     plt.show()
# 
# interact(plot_func, freq=(1, 10, 0.5))

# ==============================================
# 8. ANIMATIONS
# ==============================================
print("\n=== 8. Animations ===\n")

# Basic animation (requires FuncAnimation)
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots(figsize=(8, 4))
x = np.linspace(0, 2*np.pi, 100)
line, = ax.plot(x, np.sin(x))

def animate(i):
    line.set_ydata(np.sin(x + i/10))
    return line,

ani = FuncAnimation(fig, animate, frames=100, interval=50, blit=True)
plt.title("Sine Wave Animation")
plt.show()

# To save animation:
# ani.save('animation.mp4', writer='ffmpeg', fps=30)

# ==============================================
# 9. SPECIALIZED VISUALIZATIONS
# ==============================================
print("\n=== 9. Specialized Visualizations ===\n")

# Error bars
x = np.arange(1, 6)
y = np.array([3, 7, 2, 5, 9])
yerr = np.array([0.5, 1, 0.7, 1.2, 0.9])

plt.figure(figsize=(8, 4))
plt.errorbar(x, y, yerr=yerr, fmt='o', capsize=5, capthick=2)
plt.title("Error Bar Plot")
plt.show()

# Stem plot
plt.figure(figsize=(8, 4))
plt.stem(x, y, linefmt='b-', markerfmt='bo', basefmt='r-')
plt.title("Stem Plot")
plt.show()

# Fill between
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

plt.figure(figsize=(8, 4))
plt.fill_between(x, y1, y2, color='gray', alpha=0.3)
plt.plot(x, y1, 'b-')
plt.plot(x, y2, 'r-')
plt.title("Fill Between Plot")
plt.show()

# Stack plot
days = np.arange(1, 6)
work = [8, 8, 8, 8, 6]
sleep = [7, 7, 7, 7, 8]
leisure = [9, 9, 9, 9, 10]

plt.figure(figsize=(8, 4))
plt.stackplot(days, work, sleep, leisure, 
              labels=['Work', 'Sleep', 'Leisure'])
plt.legend(loc='upper left')
plt.title("Stack Plot")
plt.xlabel("Day")
plt.ylabel("Hours")
plt.show()

# ==============================================
# 10. REAL-WORLD APPLICATIONS
# ==============================================
print("\n=== 10. Real-World Applications ===\n")

# Financial plot
import matplotlib.dates as mdates
from datetime import datetime

dates = [datetime(2023, 1, 1), datetime(2023, 2, 1), 
         datetime(2023, 3, 1), datetime(2023, 4, 1)]
prices = [100, 115, 90, 125]

plt.figure(figsize=(10, 5))
plt.plot(dates, prices, marker='o')

# Format x-axis for dates
ax = plt.gca()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax.xaxis.set_major_locator(mdates.MonthLocator())
plt.gcf().autofmt_xdate()

plt.title("Stock Price Over Time")
plt.ylabel("Price ($)")
plt.grid(True)
plt.show()

# Geographic plot (pseudo-code - would need basemap/cartopy for real maps)
# from mpl_toolkits.basemap import Basemap
# 
# plt.figure(figsize=(10, 6))
# m = Basemap(projection='mill', llcrnrlat=20, urcrnrlat=50,
#             llcrnrlon=-130, urcrnrlon=-60, resolution='c')
# m.drawcoastlines()
# m.drawcountries()
# m.drawstates()
# 
# # Plot city locations
# lats = [40.71, 34.05, 41.88]  # NYC, LA, Chicago
# lons = [-74.01, -118.24, -87.63]
# x, y = m(lons, lats)
# m.plot(x, y, 'ro', markersize=10)
# 
# plt.title("US Cities")
# plt.show()

print("\nAll matplotlib examples completed successfully!")
