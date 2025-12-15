# General imports for plotting
import matplotlib.pyplot as plt
import numpy as np  # (currently not used, but can be useful for future extensions)

# Import for font handling and axis tick spacing
from matplotlib import font_manager
from matplotlib.ticker import MultipleLocator

# Reload the Matplotlib font manager (ensures all fonts are detected)
import matplotlib.font_manager
matplotlib.font_manager._load_fontmanager(try_read_cache=False)

# Optional: Import a custom font (currently commented out)

# If you want to use a specific font, you can activate and adjust these lines.
# font_path = '/usr/share/fonts/opentype/urw-base35/NimbusSans-Regular.otf'
# font_manager.fontManager.addfont(font_path)
# plt.rcParams['font.family'] = 'sans-serif'
# plt.rcParams['font.sans-serif'] = ['Nimbus Sans']
# plt.rcParams['font.size'] = 18

# General plot for a 2D Diagram: plot_data_2D(Data X-axis, Data Y-axis, Curve Name)
# This function creates a 2D x-y plot and saves it as a PNG image.
#
# Parameters:
#   title     → Title of the plot and name of the output file (without .png)
#   x_axis_1  → Values for the x-axis (e.g., stress)
#   y_axis_1  → Values for the y-axis (e.g., cross-section height)
#   name_1    → Name of the curve (for legend, currently commented)
def plot_data_2D(title, x_axis_1, y_axis_1, name_1):

    # Create a new figure with a specific size (in inches)
    fig = plt.figure(figsize=(7.5, 7.5))

    # Add axes to the figure:
    # [left position, bottom position, width, height] in relative coordinates (0–1)
    ax = fig.add_axes([0.15, 0.15, 0.8, 0.8])

    # Axis Tick Settings
    # Set spacing for major ticks:
    ax.xaxis.set_major_locator(MultipleLocator(10))   # x-axis ticks every 10 units
    ax.yaxis.set_major_locator(MultipleLocator(50))   # y-axis ticks every 50 units

    # Axis limits
    ax.set_xlim(-70, 70)       # x-axis range
    ax.set_ylim(300, -300)     # y-axis reversed (top = +300, bottom = -300)

    # Style of axis ticks (thickness, length, label size)
    ax.tick_params(which='major', width=0.5, length=10, labelsize=18)

    # Grid Settings
    ax.grid(linestyle="solid", linewidth=0.5, color='.25', zorder=-10)

    # Plot the main data curve
    ax.plot(
        x_axis_1,
        y_axis_1,
        c='k',                # black line
        ls='solid',           # solid line
        lw=1.5,               # line width
        label=str(name_1)     # curve name (legend currently disabled)
    )

    # Example for additional curves (currently commented)
    # ax.plot(x_axis_2, y_axis_2, c='k', ls=(0,(5,10)), lw=1.0, label=str(name_2))
    # ax.plot(x_axis_3, y_axis_3, c='k', ls='dotted', lw=1.0, label=str(name_3))
    # ax.plot(x_axis_4, y_axis_4, c='r', ls='solid', lw=1.5, label=str(name_4))

    # Axis Labels
    # ax.set_title(title, fontsize=20, verticalalignment='bottom')  # optional title
    ax.set_xlabel("Stress [N/mm²]")          # x-axis label
    ax.set_ylabel("Cross-section height [mm]")  # y-axis label

    # Legend (currently disabled)
    # If you want to show a legend for multiple curves, enable these lines:
    # legend = ax.legend(loc="lower right", prop=font)
    # legend.get_frame().set_facecolor('white')
    # legend.get_frame().set_alpha(1)
    # legend.get_frame().set_edgecolor('black')

    # Save the plot as a PNG image
    plt.savefig(str(title) + ".png", dpi=300, bbox_inches='tight')

    # Show the plot on screen (disabled because we only save the image here)
    # plt.show()
