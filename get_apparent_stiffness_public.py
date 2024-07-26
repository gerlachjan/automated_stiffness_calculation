# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 23:57:48 2023

@author: Gerlach
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import math


import matplotlib.font_manager as fm
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from scipy.signal import savgol_filter
from scipy.optimize import curve_fit

#==============================================================================
# GENERAL SETTINGS
#==============================================================================

#%matplotlib qt

plt.close('all')

#colors
iulblue = np.array([55, 96, 146])/256
c1 = np.array([127, 127, 127])/256      #grey
c2 = np.array([222, 0, 0])/256          #red
c3 = np.array([0, 176, 80])/256         #green
c4 = np.array([210, 210, 210])/256      #light grey
c5 = np.array([238, 127, 0])/256        #orange
c6 = np.array([240, 182, 0])/256        #creame yellow

c7 = np.array([0, 104, 178])/256        #TRR blue
c8 = np.array([250, 174, 40])/256        #TRR orange
c9 = np.array([118, 183, 26])/256        #TRR green

# if True: Enables additional visualizations for debugging purpose
debug_mode = False
#==============================================================================
# FUNCTIONS
#==============================================================================


# def plot2D_IUL_style(fig, ax, xlim=False, ylim=False, xlabel='xlabel', ylabel='ylabel', fontsize=16, loc='lower right', safe_plot=False, legend=False, noaxis=False, infobox_str=None, infobox_pos=[0.63, 0.935], count=0, plot_name = ''):
    
#     """
#     Apply a consistent IUL style to a 2D matplotlib plot.

#     Parameters:
#         fig (matplotlib.figure.Figure): the figure object to apply the style to.
#         ax (matplotlib.axes.Axes): the axis object to apply the style to.
#         xlim (tuple, optional): the limits for the x-axis. Defaults to False.
#         ylim (tuple, optional): the limits for the y-axis. Defaults to False.
#         xlabel (str, optional): the label for the x-axis. Defaults to 'xlabel'.
#         ylabel (str, optional): the label for the y-axis. Defaults to 'ylabel'.
#         fontsize (int, optional): the font size to use for the axis labels. Defaults to 16.
#         loc (str, optional): the location for the legend. Defaults to 'lower right'.
#         safe_plot (bool, optional): whether to save the plot to a file. Defaults to False.
#         legend (bool, optional): whether to show a legend. Defaults to False.
#         noaxis (bool, optional): whether to show the axis. Defaults to False.
#         infobox_str (str, optional): a string to display in a box on the plot. Defaults to None.
#         infobox_pos (list, optional): the position for the info box as a list of [x,y] coordinates. Defaults to [0.63, 0.935].
#         count (int, optional): a number to append to the plot file name if safe_plot is True. Defaults to None.

#     Returns:
#         None
#     """
    
#     ax.tick_params(width=2, length=4)
#     ax.set_xlim(xlim)
#     ax.set_ylim(ylim)
#     ax.grid(True, linewidth=1)
#     # ax.minorticks_on()
#     ax.set_axisbelow(True)
#     ax.spines['left'].set_linewidth(2)
#     ax.spines['bottom'].set_linewidth(2)
#     ax.set_xlabel(xlabel, fontsize=fontsize, fontweight='bold', family='Arial')
#     ax.set_ylabel(ylabel, fontsize=fontsize, fontweight='bold', family='Arial')
   
#     font = fm.FontProperties(family='Arial', size=fontsize)
#     font_legend = fm.FontProperties(family='Arial', size=fontsize-5)
#     plt.rcParams["mathtext.fontset"] = "cm"
    
#     ax.xaxis.set_tick_params()
#     ax.yaxis.set_tick_params()
    
#     for label in ax.get_xticklabels():
#         label.set_fontproperties(font)
#     for label in ax.get_yticklabels():
#         label.set_fontproperties(font)
    
#     if legend:
#         ax.legend(prop=font_legend, facecolor='white', fancybox=False, framealpha=1, edgecolor='black', loc=loc)
    
#     if noaxis:
#         plt.axis('off')
    
#     if infobox_str:
#         props = dict(boxstyle='square', facecolor='white', alpha=1)
#         ax.text(infobox_pos[0], infobox_pos[1], infobox_str, transform=ax.transAxes, fontsize=fontsize-5, family='Arial', verticalalignment='center', bbox=props)
    
#     fig.set_size_inches(6.30, 3.54)
#     # fig.set_size_inches(3.25, 3.54)
#     fig.tight_layout(pad=0.5, w_pad=0.5, h_pad = 1.0)
                     
#     if safe_plot:
#         label = f'{count:03}'
#         fig.savefig('plots/'+plot_name+label+'.png', dpi=300)
#         fig.savefig('plots/'+plot_name+label+'.tiff', dpi=300)
#         fig.savefig('plots/'+plot_name+label+'.eps', dpi=300)
#         fig.savefig('plots/'+plot_name+label+'.svg', dpi=300)
        
def iul_plot_style(fig,ax,ax2=False, xlim = False, ylim=False,ylim2=None, xlabel='xlabel',ylabel='ylabel',ylabel2 = None, fontsize = 16, loc = 'upper right', safe_plot=False, plot_name='', legend=False, noaxis=False, infobox_string = None, infobox_pos =[0.63, 0.935]):
    ax.tick_params(width=2, length=4)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.grid(True, linewidth=1)
    ax.minorticks_on()
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_axisbelow(True)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.set_xlabel(xlabel,fontsize=fontsize, fontweight='bold', family='Arial')
    ax.set_ylabel(ylabel,fontsize=fontsize,fontweight='bold', family='Arial')
    # ax.set_xlabel(xlabel,fontsize=fontsize, family='Arial')
    # ax.set_ylabel(ylabel,fontsize=fontsize, family='Arial')
    # fig.suptitle('Stress-strain curve of extruded component',fontsize=fontsize,fontweight='bold', family='Arial')
    
    if ax2:
      
        ax2.tick_params(width=2, length=4)
        ax2.set_xlim(xlim)
        ax2.set_ylim(ylim2)
        ax2.grid(False)
        ax2.minorticks_on()
        ax2.xaxis.set_minor_locator(plt.NullLocator())
        ax2.set_axisbelow(True)
        ax2.spines['left'].set_linewidth(2)
        ax2.spines['bottom'].set_linewidth(2)
        ax2.set_xlabel(xlabel,fontsize=fontsize, fontweight='bold', family='Arial')
        ax2.set_ylabel(ylabel2,fontsize=fontsize,fontweight='bold', family='Arial')
        
    
    font = fm.FontProperties(family='Arial',size = fontsize)
    font_legend = fm.FontProperties(family='Arial',size = fontsize-2)
    plt.rcParams["mathtext.fontset"] = "cm"
    
    ax.xaxis.set_tick_params()
    ax.yaxis.set_tick_params()
    
    if ax2:
        ax2.yaxis.set_tick_params()
        
    
    if noaxis == True:
        ax.set_yticklabels([])

    for label in ax.get_xticklabels():
        label.set_fontproperties(font)
    for label in ax.get_yticklabels():
        label.set_fontproperties(font)
        
    if ax2:
        for label in ax2.get_xticklabels():
            label.set_fontproperties(font)
        for label in ax2.get_yticklabels():
            label.set_fontproperties(font)
    
    if legend and ax2:
        # Collect handles and labels from both axes
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()

        # Combine them
        handles = handles1 + handles2
        labels = labels1 + labels2

        # Create a single legend
        ax2.legend(handles, labels, prop=font_legend, facecolor='white', fancybox=False, framealpha=1, edgecolor='black', loc=loc,labelspacing = 0.05)
    
    if legend and not ax2:
        ax.legend(prop=font_legend,facecolor='white', fancybox=False, framealpha=1,edgecolor='black', loc=loc,labelspacing = 0.05)
    
    if not infobox_string == None:
        props = dict(boxstyle='square', facecolor='white', alpha=1)
        # textstr = '\n'.join((r'$\mu=%.2f$' % (0.3, ),r'$\mathrm{E}=%d$MPa' % (210000, ),r'$\sigma_f=%.2f$MPa' % (550, )))
        textstr = infobox_string
        # place a text box in upper left in axes coords
        # ax.text(0.63, 0.935, textstr, transform=ax.transAxes, fontsize=fontsize-5, family='Arial', verticalalignment='center', bbox=props)
        # ax.text(0.535, 0.935, textstr, transform=ax.transAxes, fontsize=fontsize-5, family='Arial', verticalalignment='center', bbox=props)
        ax.text(infobox_pos[0],infobox_pos[1], textstr, transform=ax.transAxes, fontsize=fontsize-3, family='Arial', verticalalignment='center', bbox=props)
    
    
    fig.set_size_inches(6.30, 3.54)
    # fig.set_size_inches(3.25, 3.54)
    fig.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
    
    if safe_plot:
        fig.savefig('plots/'+plot_name +'plot.png', dpi=300)
        fig.savefig('plots/'+plot_name +'plot.tiff', dpi=300)
        fig.savefig('plots/'+plot_name +'plot.eps', dpi=300)
        fig.savefig('plots/'+plot_name +'plot.svg', dpi=300)

def get_expdata_from_excel_sheet(folder_path = 'data'):

    folder_path = folder_path
    
    data = {}
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.xlsx'):
            excel_file = pd.ExcelFile(os.path.join(folder_path, file_name))
        else:
            continue
        sample_sheet = excel_file.parse('Ergebnisse')
        sample_names = sample_sheet.iloc[1:, [0, 1]].values
        
        for sheet_name in excel_file.sheet_names:
            if sheet_name in sample_names[:,0]:
                foo_index = np.where(sample_names[:, 0] == sheet_name)
                if foo_index[0].size > 0:
                    sample_name = sample_names[foo_index, 1][0][0]
                sheet_data = excel_file.parse(sheet_name, header=1, skiprows=[2]) # setting second row as header
                data[sample_name] = sheet_data
                #data[sample_name] = excel_file.parse(sheet_name).iloc[2:,:].values
                
    return data

def split_data_unloading(x,y,tol):
    
    dx = np.diff(x)
    dy = np.diff(y)
    
    distance = np.sqrt(dx**2 + dy**2)

    idx = np.concatenate(np.where(dx > tol)) + 1 #first indicies of each unloading step. 
    
    # #TODO: in some cases the first cycle is not recognized
    idx = np.insert(idx,0,0)
   
    num_unloading_steps = len(idx)
    
    
    x_split = []
    y_split = []
    for i in range(num_unloading_steps):
        if i == num_unloading_steps - 1:
            x_split.append(x[idx[i]:])
            y_split.append(y[idx[i]:])
        else:
            x_split.append(x[idx[i]:idx[i+1]])
            y_split.append(y[idx[i]:idx[i+1]])
            
    if debug_mode:
        fig, ax = plt.subplots()
        ax.scatter(x[:-1],dx, s=2)
        plt.title('dx value to determin tol value (unloading split)')    
        
    return x_split, y_split


def split_data_loading(x,y,tol):
    
    dx = np.diff(x)
    dy = np.diff(y)
    
    distance = np.sqrt(dx**2 + dy**2)

    idx = np.concatenate(np.where(dx < tol)) + 1 #first indicies of each (re-)loading step!!!! not initial loading
    
    num_loading_steps = len(idx) + 1

    x_split = []
    y_split = []
    
    for i in range(num_loading_steps):
        if i == 0:
            x_split.append(x[:idx[i]])
            y_split.append(y[:idx[i]])
        elif i == num_loading_steps - 1:
            x_split.append(x[idx[i-1]:])
            y_split.append(y[idx[i-1]:])
        else:
            x_split.append(x[idx[i-1]:idx[i]])
            y_split.append(y[idx[i-1]:idx[i]])
            
    if debug_mode:
        fig, ax = plt.subplots()
        ax.scatter(x[:-1],dx, s=2)
        plt.title('dx value to determin tol value (loading split)')
                   
    return x_split, y_split

def shift_data_to_other_list(nshift, list1, list2):
    
    for idx in range(len(list1)-1):
        temp_list1 = list1[idx][:-nshift]
        temp_list2 = np.concatenate((list1[idx][-nshift:],list2[idx]))
    
    
        list1[idx] = temp_list1
        list2[idx] = temp_list2
        
    return list1, list2


def get_interpolated_data(x,y,ndata):
    
    """
    Generate the interpolated dataset using linear interpolation based on adjacent points

    """
    
    interpolated_data = []
    for i in range(len(y) - 1):
        # Extract x and y values for the two points
        x0, y0 = x[i], y[i]
        x1, y1 = x[i + 1], y[i+1]

        # Calculate the interpolated values
        for j in range(ndata):
            alpha = j / ndata
            x_new = (1 - alpha) * x0 + alpha * x1
            y_new = (1 - alpha) * y0 + alpha * y1
            interpolated_data.append([x_new, y_new])
        
    interpolated_data = np.array(interpolated_data)
    x_new = interpolated_data[:,0]
    y_new = interpolated_data[:,1]

    return x_new, y_new
    

def get_intersection_points_for_curves(loading_curve, unloading_curve):
    
    """
    Iterates through the two curves and returns the intersection point (as a tuple)

    """
    intersection_points = []

    for i in range(len(loading_curve) - 1):
        for j in range(len(unloading_curve) - 1):
            intersection_point = get_intersecting_point_from_points(loading_curve[i], loading_curve[i+1], unloading_curve[j], unloading_curve[j+1])
            
            if intersection_point:
                intersection_points.append(intersection_point)

    return intersection_points


def get_intersecting_point_from_points(point_a, point_b, point_c, point_d):
    
    """
    Returns the intersection point of two lines as a tuple

    """
    
    # Checks if point_a and point_b lie on opposite sides of the line made by point_c and point_d
    if (is_ccw(point_a, point_c, point_d) != is_ccw(point_b, point_c, point_d) and is_ccw(point_a, point_b, point_c) != is_ccw(point_a, point_b, point_d)):
        # Calculates the intersection point using the slopes and y-intercepts
        slope_AB = (point_b[1] - point_a[1]) / (point_b[0] - point_a[0])
        y_intercept_AB = point_a[1] - slope_AB * point_a[0]

        slope_CD = (point_d[1] - point_c[1]) / (point_d[0] - point_c[0])
        y_intercept_CD = point_c[1] - slope_CD * point_c[0]

        x_intersect = (y_intercept_CD - y_intercept_AB) / (slope_AB - slope_CD)
        y_intersect = slope_AB * x_intersect + y_intercept_AB

        # Checks if the intersection point is within both line segments
        if is_on_segment(point_a, (x_intersect, y_intersect), point_b) and is_on_segment(point_c, (x_intersect, y_intersect), point_d):
            return (x_intersect, y_intersect)

    return None


# Returns true if triangle made by point_d, point_e, point_f is counterclockwise
def is_ccw(point_d, point_e, point_f):
    return (point_f[1] - point_d[1]) * (point_e[0] - point_d[0]) > (point_e[1] - point_d[1]) * (point_f[0] - point_d[0])


# Returns true if point_q lies on the line segment made by point_p and point_r
def is_on_segment(point_p, point_q, point_r):
    return (point_q[0] <= max(point_p[0], point_r[0]) and point_q[0] >= min(point_p[0], point_r[0]) and
            point_q[1] <= max(point_p[1], point_r[1]) and point_q[1] >= min(point_p[1], point_r[1]))


# Returns the point closest to the origin
def get_closest_point(point_a, point_b):
    # Calculates the distance for each point from the origin
    distance_a = math.sqrt(point_a[0]**2 + point_a[1]**2)
    distance_b = math.sqrt(point_b[0]**2 + point_b[1]**2)

    # Compares distances and determines which point is closest
    if distance_a < distance_b:
        closest_point = point_a
    else:
        closest_point = point_b
        
    return closest_point



def calculate_slope(point_a, point_b):
    
    """
    Returns the slope of a line given by two points (tuples)

    """
    try:
        slope = (point_b[1] - point_a[1]) / (point_b[0] - point_a[0])
        return abs(slope)
    except ZeroDivisionError:
        return None
    
def fit_function(x,m, b):
    return m*x + b

def regression(x,y):
    popt, pcov = curve_fit(fit_function, x, y)
    return popt

def yoshida_uemori_model(eps_p,E0,Ea,psi):

    Eu = E0 - (E0-Ea)*(1-np.exp(-psi*eps_p))

    return Eu
    
def calc_effective_strain_stress(u,force,l0,a0, necking = False):
    
    """
    Calcultes effective stress-strain curve
    
    l0: extensometer length
    a0: initial cross section of specimen

    """
    
    u_temp = []
    force_temp = []
    for i in range(len(u)):
        if u[i] > 0:
            u_temp.append(u[i])
            force_temp.append(force[i])
    
    u_temp = np.array(u_temp)
    force_temp = np.array(force_temp)
    
    idx_maxF = np.argmax(force_temp)
    
    eps = u_temp/l0
    phi = np.log(1 + eps)
    
    sig = force_temp/a0
    sig_true = (1 + eps)*sig
    
    if necking:
        return phi, sig_true
    else:
        return phi[:idx_maxF],sig_true[:idx_maxF]
    

    

def get_apparent_stiffness(xs,ys):
    

    # filter or smooth raw data for better determination of loading and unloading part 
    xhat = savgol_filter(xs, 5, 2)
    yhat = savgol_filter(ys, 5, 2) 
    
    # calculation of derivative to determine transition point between loading and unloading
    dxhat = np.diff(xhat)
    dyhat = np.diff(yhat)
    
    loading_x = []
    loading_y = []
    unloading_y = []
    unloading_x = []
    
    # split data into loading and unloading based on the change in the derivative 
    for idx,value in enumerate(dxhat):
        if value > 0:
            loading_x.append(xhat[idx+1])
            loading_y.append(yhat[idx+1])
        else:
            unloading_x.append(xhat[idx+1])
            unloading_y.append(yhat[idx+1])
      
    """
    Seperate loading and unloading in separate arrays for each loading /unloading step
    Remark: tol = -0.05 (loading) and tol = 0.05 (unloading) are values which have to be manualy choosen. Depending on the data values, it might be different
    """
    #TODO
    # x_split_loading, y_split_loading = split_data_loading(np.array(loading_x), np.array(loading_y), -0.05)
    # x_split_unloading, y_split_unloading = split_data_unloading(np.array(unloading_x), np.array(unloading_y), 0.05)
    
    x_split_loading, y_split_loading = split_data_loading(np.array(loading_x), np.array(loading_y), -0.0002)
    x_split_unloading, y_split_unloading = split_data_unloading(np.array(unloading_x), np.array(unloading_y), 0.0002)
      
    """
    The exact data split at the transition point between loading and unloading is not possible with the above criterion.
    Therefore the last "nshift" datapoints are moved from loading list n to unloading list n+1
    """
    x_split_loading, x_split_unloading =  shift_data_to_other_list(5,x_split_loading, x_split_unloading)
    y_split_loading, y_split_unloading =  shift_data_to_other_list(5,y_split_loading, y_split_unloading)
    
    if debug_mode:
        fig, ax = plt.subplots()
        for i in range(len(x_split_loading)):
            ax.plot(x_split_loading[i],y_split_loading[i],label='Loading')
        for i in range(len(x_split_unloading)):
            ax.plot(x_split_unloading[i],y_split_unloading[i],linestyle = ':', label='Unloading')
        ax.legend()
        ax.set_title('Split of data in loading and unloading as well as in separate cycles')
        
    """
    Calculation of stiffness based on elastic loading curve

    """
    #TODO: Currently the threshhold for data consideration of loading stiffness calculation has to be defined manually
    # ylim_for_stiffness = 6 #LDC of notched R10_t5 force in kN
    # ylim_for_stiffness_first_loading = 6 #LDC of notched R10_t5 force in kN
    
    ylim_for_stiffness = 400 #stress strain of simple tension
    ylim_for_stiffness_first_loading = 180 #stress strain of simple tension
    
    
    if debug_mode:
        fig, ax = plt.subplots()
        ax.plot(xhat, yhat)
    
    dict_loading_stiffness = {}
    for i in range(len(x_split_loading)):
        if i == 0:
            idx_ylim = np.where(y_split_loading[i] >  ylim_for_stiffness_first_loading)[0][0]
            params = regression(x_split_loading[i][:idx_ylim], y_split_loading[i][:idx_ylim])
            dict_loading_stiffness[i] = params[0]
        else:
            idx_ylim = np.where(y_split_loading[i] >  ylim_for_stiffness)[0][0]
            params = regression(x_split_loading[i][:idx_ylim], y_split_loading[i][:idx_ylim])
            dict_loading_stiffness[i] = params[0]
        if debug_mode:
            ax.plot(x_split_loading[i], fit_function(x_split_loading[i],params[0],params[1]),color = 'black', linewidth=1,alpha=1,linestyle='--')
    
    list_loading_stiffness = list(dict_loading_stiffness.values())
    
    if debug_mode:
        iul_plot_style(fig, ax,xlim= [0,1.1*max(xhat)],ylim = [0,1.1*max(yhat)], xlabel=r'x', ylabel= r'y',safe_plot=False,noaxis=False,legend = False, plot_name = 'debugging')
    
    """
    Calculation of apparent stiffness for each loading & unloading cylcle.
    The values are stored in a dictionary called dict_apparent_stiffness.
    """
    
    dict_apparent_stiffness = {}
    dict_mean_displacement_cycle = {}
    
    list_intersection_points = []
    list_intersection_point_Fmax = []
    list_closest_point = []
    
    for i in range(len(x_split_unloading)):
        # Creates a list of tuples for the curves
        loading_curve = list(zip(x_split_loading[i+1], y_split_loading[i+1]))
        unloading_curve = list(zip(x_split_unloading[i], y_split_unloading[i]))
        
        intersection_points = get_intersection_points_for_curves(loading_curve, unloading_curve)
        list_intersection_points.append(intersection_points)
        
        #if multiple intersection points due to data noise are existing, find the intersection point with the max force value
        if not intersection_points: # only relevant if data has no intersection point, this is an indicator for mistake experimental data
            print("Intersection point is not existing and is set to (0,0)")
            intersection_point_Fmax = (0,0)
            idx_intersection_point_Fmax = 0
        else:
            intersection_point_Fmax = max(intersection_points, key=lambda x: x[1])
            idx_intersection_point_Fmax = intersection_points.index(intersection_point_Fmax)
        
        list_intersection_point_Fmax.append(intersection_point_Fmax)
        closest_point = get_closest_point(loading_curve[0], unloading_curve[len(unloading_curve)-1])
        list_closest_point.append(closest_point)
        
        
        mean_displacement_cycle =  closest_point[0] + 0.5*(intersection_point_Fmax[0] - closest_point[0])  
        dict_mean_displacement_cycle[i] = mean_displacement_cycle
            
        apparent_stiffness = calculate_slope(intersection_point_Fmax, closest_point)
        dict_apparent_stiffness[i] = apparent_stiffness
        
    list_apparent_stiffness = list(dict_apparent_stiffness.values())
    list_mean_displacement_cycles = list(dict_mean_displacement_cycle.values())
    list_mean_displacement_cycles_plus_origin = [0] + list_mean_displacement_cycles
    
    
    if debug_mode:
        
        """
        Visualization of relevant calculation points for apparent stiffness

        """
        fig,ax = plt.subplots()
        ax.plot(xhat, yhat, zorder=1)
        for i,j in zip(list_intersection_points, list_closest_point):
            ax.scatter(i[idx_intersection_point_Fmax][0],i[idx_intersection_point_Fmax][1],color ='red',s=20,zorder=2)
            ax.scatter(j[0],j[1], color='red', s=20, zorder=2)
            ax.plot([i[idx_intersection_point_Fmax][0], j[0]], [i[idx_intersection_point_Fmax][1], j[1]], linestyle = '--', color='grey')  # Drawing a line between i and j
        ax.set_xlabel('Displacement in mm')
        ax.set_ylabel('Force in N')
        ax.grid()
        
        iul_plot_style(fig, ax,xlim= [0,1.1*max(xhat)],ylim = [0,1.1*max(yhat)], xlabel=r'x', ylabel= r'y',safe_plot=False,noaxis=False,legend = False, plot_name = 'debugging')
        
    return list_apparent_stiffness,list_loading_stiffness,list_mean_displacement_cycles,list_mean_displacement_cycles_plus_origin,list_intersection_point_Fmax, list_closest_point

#==============================================================================
# MAIN
#==============================================================================

if __name__ == "__main__":
    
    # # data import
    # df_monoton      = pd.read_csv('LDC_DB_R10_t5_05_tactile.txt', delimiter=';', header=None, names=['Displacement', 'Force'])
    # x_monoton      = df_monoton['Displacement'] 
    # y_monoton   = df_monoton['Force']/1000
    
    
    # df = get_expdata_from_excel_sheet(folder_path = '2023-09-03_DP800_zyklische_Zugversuch')
    
    
    # # plot initialization 
    # fig1,ax1 = plt.subplots() # Force/displacement or stress/strain curve 
    # axins = inset_axes(ax1, width="30%", height="60%", loc=4)  # zoom into one unloading-reloading cylce within 
    
    # fig2,ax2 = plt.subplots()   # Evolution of stiffness
    
    # for sheet_name in df.keys():
        
    #     data = df[sheet_name]
    #     x = data['Dehnung'].values + 0.009575 #'Dehnung' is not strain here (wrong label used during export). Here it is the displacement!
    #     y = data['Standardkraft'].values/1000
        
    #     # Visulalization of cyclic force/displacement or stress/strain curve 
    #     ax1.plot(x, y, label = sheet_name, color=c7, alpha = 0.6)
    #     axins.plot(x, y, linewidth =0.75, label=sheet_name, color=c7, alpha=0.6, zorder = 1)
        
        
    #     list_apparent_stiffness,list_loading_stiffness,list_mean_displacement_cycle,list_mean_displacement_cycle_plus_origin,list_intersection_point_Fmax, list_closest_point = get_apparent_stiffness(x, y)
        
    #     if sheet_name == 'R10_t5_5':
    #         # Zoom plot for certain cycle of specimen with label "sheet_name"
    #         axins.plot([list_intersection_point_Fmax[2][0], list_closest_point[2][0]], [list_intersection_point_Fmax[2][1], list_closest_point[2][1]], linewidth = 1.2, linestyle = '--', color='white')
    #         axins.plot([list_intersection_point_Fmax[2][0], list_closest_point[2][0]], [list_intersection_point_Fmax[2][1], list_closest_point[2][1]],linewidth = 1, linestyle = '--', color='black')
    #         axins.scatter(list_intersection_point_Fmax[2][0],list_intersection_point_Fmax[2][1],color ='white',s=35,zorder=2)
    #         axins.scatter(list_intersection_point_Fmax[2][0],list_intersection_point_Fmax[2][1],color ='black',s=20,zorder=2)
    #         axins.scatter(list_closest_point[2][0],list_closest_point[2][1], color='white', s=35, zorder=2)
    #         axins.scatter(list_closest_point[2][0],list_closest_point[2][1], color='black', s=20, zorder=2)
            
    
    #     print(f"Apparent stiffness ({sheet_name}):")
    #     print("\n".join(map(str, list_apparent_stiffness)))
    
    #     # Visualization of apparent stiffness evolution
    #     ax2.plot(list_mean_displacement_cycle, list_apparent_stiffness,linewidth = 2,  marker='o', markersize=4,markerfacecolor='white',fillstyle='full',color='white', alpha = 1, label = sheet_name, zorder = 1)  
    #     ax2.plot(list_mean_displacement_cycle, list_apparent_stiffness,linewidth = 1, marker='o',markersize=4, markerfacecolor=c7,fillstyle='full',color=iulblue, alpha = 0.6, label = sheet_name, zorder = 2)  
    
    # # Visulalization of monoton force/displacement or stress/strain curve 
    # ax1.plot(x_monoton, y_monoton, linewidth = 2, color = c8,label = sheet_name)
    # axins.plot(x_monoton, y_monoton, color = c8,label = sheet_name)
    
    # # Set the axis limits for the zoom plot
    # axins.set_xlim(0.395, 0.5)
    # axins.set_ylim(1, 12.5)

    # # Hide the tick labels
    # axins.set_xticks([])
    # axins.set_yticks([])

    # # Draw a box indicating where the inset is zooming from
    # mark_inset(ax1, axins, loc1=1, loc2=3, fc="none", ec="0.5")
    
    
    
    
    
    # #ax2.plot(list_mean_displacement_cycle, dfdu_sim_list[1:],linewidth = 1, marker='o',markersize=4, markerfacecolor=c7,fillstyle='full',color='green', alpha = 0.6, label = sheet_name, zorder = 2)  
    
    # iul_plot_style(fig2, ax2,xlim=[0,1.8], ylim=[90,130],xlabel='Displacement in mm', ylabel= 'Apparent stiffness in kN/mm',safe_plot=True,noaxis=False,legend = False, plot_name = 'K_ref')
    # iul_plot_style(fig1, ax1,xlim=[0,1.8], ylim=[0,14],xlabel=r'Displacement in mm', ylabel= r'Force in kN',safe_plot=True,noaxis=False,legend = False, plot_name = 'LDC_ref_monoton')


    df2 = get_expdata_from_excel_sheet(folder_path = '2022-12-09_DP800_Zyklische_Zugversuche')
    label_of_interests = ['Simple Tension 4','Simple Tension 5','Simple Tension 6'] 
    

    fig1,ax1 = plt.subplots() # Stress/strain curve 
    axins1 = inset_axes(ax1, width="30%", height="60%", loc=4)  # zoom into one unloading-reloading cylce within 
    fig2,ax2 = plt.subplots()   # Evolution of stiffness
    for label in label_of_interests:
        data2 = df2[label]
        
        x2 = data2['Dehnung'].values # displacement in mm
        y_temp = data2['Standardkraft'].values # force in kN
        
        y2 = y_temp + abs(y_temp[0]) #Due to missed reset of inital force in tensile test machine, the LDC curve has to be moved manually to zero force level. Might not be necessary for other data sets
        
        eps, sig = calc_effective_strain_stress(x2,y2,l0 = 66,a0 = 30, necking = False)
        
        list_apparent_stiffness,list_loading_stiffness,list_mean_displacement_cycle,list_mean_displacement_cycles_plus_origin,list_intersection_point_Fmax, list_closest_point = get_apparent_stiffness(eps, sig)
        
        # Visulalization of cyclic force/displacement or stress/strain curve 
        ax1.plot(eps, sig, label = label, color=c7, alpha = 0.6)
        axins1.plot(eps, sig, linewidth =0.75, label=label, color=c7, alpha=0.6, zorder = 1)
        
        if label == 'Simple Tension 4':
            # Zoom plot for certain cycle of specimen with label "label"
            axins1.plot([list_intersection_point_Fmax[2][0], list_closest_point[2][0]], [list_intersection_point_Fmax[2][1], list_closest_point[2][1]], linewidth = 1.2, linestyle = '--', color='white')
            axins1.plot([list_intersection_point_Fmax[2][0], list_closest_point[2][0]], [list_intersection_point_Fmax[2][1], list_closest_point[2][1]],linewidth = 1, linestyle = '--', color='black')
            axins1.scatter(list_intersection_point_Fmax[2][0],list_intersection_point_Fmax[2][1],color ='white',s=35,zorder=2)
            axins1.scatter(list_intersection_point_Fmax[2][0],list_intersection_point_Fmax[2][1],color ='black',s=20,zorder=2)
            axins1.scatter(list_closest_point[2][0],list_closest_point[2][1], color='white', s=35, zorder=2)
            axins1.scatter(list_closest_point[2][0],list_closest_point[2][1], color='black', s=20, zorder=2)
            
        
        print(f"Apparent stiffness ({label}):")
        print("\n".join(map(str, list_apparent_stiffness)))
        
        print(f"Loading stiffness ({label}):")
        print("\n".join(map(str, list_loading_stiffness)))
        
        # Visualization of apparent stiffness evolution
        ax2.plot(list_mean_displacement_cycle, np.array(list_apparent_stiffness)/1000,linewidth = 2,  marker='o', markersize=4,markerfacecolor='white',fillstyle='full',color='white', alpha = 1, label = label, zorder = 1)  
        ax2.plot(list_mean_displacement_cycle, np.array(list_apparent_stiffness)/1000,linewidth = 1, marker='o',markersize=4, markerfacecolor='dimgrey',fillstyle='full',color='dimgrey', alpha = 1, label = label, zorder = 2)  
        
        ax2.plot(list_mean_displacement_cycles_plus_origin[1:], np.array(list_loading_stiffness[1:])/1000,linewidth = 2,  marker='o', markersize=4,markerfacecolor='white',fillstyle='full',color='white', alpha = 1, label = label, zorder = 1)  
        ax2.plot(list_mean_displacement_cycles_plus_origin[1:], np.array(list_loading_stiffness[1:])/1000,linewidth = 1, marker='o',markersize=4, markerfacecolor='darkgrey',fillstyle='full',color='darkgrey', alpha = 1, label = label, zorder = 2)  
        
        ax2.plot(list_mean_displacement_cycles_plus_origin[:2], np.array(list_loading_stiffness[:2])/1000, linestyle ='--', dashes=(5, 5), linewidth = 2,  marker='o', markersize=4,markerfacecolor='white',fillstyle='full',color='white', alpha = 1, label = label, zorder = 1)  
        ax2.plot(list_mean_displacement_cycles_plus_origin[:2], np.array(list_loading_stiffness[:2])/1000, linestyle ='--', dashes=(5, 5), linewidth = 1, marker='o',markersize=4, markerfacecolor='darkgrey',fillstyle='full',color='darkgrey', alpha = 1, label = label, zorder = 2)  
    
    # Set the axis limits for the zoom plot
    axins1.set_xlim(0.048, 0.054)
    axins1.set_ylim(25, 880)

    # Hide the tick labels
    axins1.set_xticks([])
    axins1.set_yticks([])

    # Draw a box indicating where the inset is zooming from
    mark_inset(ax1, axins1, loc1=1, loc2=3, fc="none", ec="0.5")    
    

        
    iul_plot_style(fig2, ax2,xlim=[0,0.3], ylim=[140,210], ylim2=[0,0.16],xlabel=r'Effective strain $\bar{\varepsilon}$', ylabel= 'Apparent Young\'s modulus in GPa',ylabel2= r'Void area fraction $D$ in %', safe_plot=True,noaxis=False,legend = False, plot_name = 'apparent_E_w_simulation')   
    iul_plot_style(fig1, ax1,xlim=[0,0.14], ylim=[0,1000],xlabel=r'Effective strain $\bar{\varepsilon}$', ylabel= r'Effective stress $\bar{\sigma}$',safe_plot=True,noaxis=False,legend = False, plot_name = 'effective_stress_strain')   
