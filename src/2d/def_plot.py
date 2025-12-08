# import general
import matplotlib.pyplot as plt
import numpy as np
# import specific functions
from matplotlib import font_manager
from matplotlib.ticker import MultipleLocator

import matplotlib.font_manager
matplotlib.font_manager._load_fontmanager(try_read_cache=False)

# import font
# font_path = '/usr/share/fonts/opentype/urw-base35/NimbusSans-Regular.otf'
# font_manager.fontManager.addfont(font_path)
# plt.rcParams['font.family'] = 'sans-serif'
# plt.rcParams['font.sans-serif'] = ['Nimbus Sans']
# plt.rcParams['font.size'] = 18

# general plot for 2D Diagramm - plot_data_2D(Data X-axis, Data Y-Axis, Name)
def plot_data_2D(title,x_axis_1,y_axis_1,name_1):
    # Create figure and axes with size
    fig = plt.figure(figsize=(7.5, 7.5))
    ax = fig.add_axes([0.15, 0.15, 0.8, 0.8])
    # Axis
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.yaxis.set_major_locator(MultipleLocator(50))

    ax.set_xlim(-70, 70)
    ax.set_ylim(300, -300)

    ax.tick_params(which='major', width=0.5, length=10, labelsize=18)

    # Grid
    ax.grid(linestyle="solid", linewidth=0.5, color='.25', zorder=-10)

    # Plot Data
    ax.plot(x_axis_1, y_axis_1, c='k', ls='solid', lw=1.5, label=str(name_1))
    # ax.plot(x_axis_2, y_axis_2, c='k', ls=(0,(5,10)), lw=1.0, label=str(name_2))
    # ax.plot(x_axis_3, y_axis_3, c='k', ls='dotted', lw=1.0, label=str(name_3))
    # ax.plot(x_axis_4, y_axis_4, c='r', ls='solid', lw=1.5, label=str(name_4))

    # Axis Naming
    # ax.set_title(title, fontsize=20, verticalalignment='bottom')
    ax.set_xlabel("Stress [N/mm²]")
    ax.set_ylabel(" Cross-section height [mm]")
    
    # Legend
    # legend = ax.legend(loc="lower right", prop=font)
    # legend.get_frame().set_facecolor('white'); legend.get_frame().set_alpha(1)
    # legend.get_frame().set_edgecolor('black')
    
    plt.savefig(str(title)+".png", dpi=300, bbox_inches='tight')
    #plt.show()

