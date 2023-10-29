import datetime
import numpy as np
import spiceypy
from sunpy.time import TimeRange
from astropy import time, units as u
from astropy.coordinates import solar_system_ephemeris
import matplotlib.pyplot as plt
import jplephem

import seaborn as sns

import os
from pathlib import Path


def main():
    print("Running solar system model")

    spice_files = ['naif0012.tls',
                   'de430.bsp',
                   'mars_iau2000_v1.tpc',
                   'pck00011.tpc']
    spiceypy.furnsh(spice_files)

    timerange = TimeRange('2020-01-01', 10 * u.year)
    global times
    times = [timerange.start.datetime]
    t = times[0]
    while t < timerange.end:
        t = t + datetime.timedelta(hours=24)
        times.append(t)

    global time_spice
    time_spice = [spiceypy.str2et(t.strftime('%Y-%m-%d %H:%M')) for t in times]

    solar_system_ephemeris.set("jpl")

    positions_earth, lightTimes_earth = spiceypy.spkezr('Earth', time_spice, 'ECLIPJ2000', 'NONE', 'Sun')
    positions_mars, lightTimes_mars = spiceypy.spkezr('Mars Barycenter', time_spice, 'ECLIPJ2000', 'NONE', 'Sun')
    positions_sun, lightTimes_earth = spiceypy.spkezr('Sun', time_spice, 'ECLIPJ2000', 'NONE', 'Sun')

    per_earth = np.array(positions_earth)[:, :3] * u.km
    global  x_earth, y_earth, z_earth
    x_earth = per_earth[:, 0].to(u.au)
    y_earth = per_earth[:, 1].to(u.au)
    z_earth = per_earth[:, 2].to(u.au)

    per_mars = np.array(positions_mars)[:, :3] * u.km
    global x_mars, y_mars, z_mars
    x_mars = per_mars[:, 0].to(u.au)
    y_mars = per_mars[:, 1].to(u.au)
    z_mars = per_mars[:, 2].to(u.au)

    per_sun = np.array(positions_sun)[:, :3] * u.km
    global x_sun, y_sun, z_sun
    x_sun = per_sun[:, 0].to(u.au)
    y_sun = per_sun[:, 1].to(u.au)
    z_sun = per_sun[:, 2].to(u.au)

    fig, ax = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(15, 10))
    ax[0,0].plot(x_earth, z_earth)
    ax[0,0].set_xlabel('x (AU)')
    ax[0,0].set_ylabel('z (AU)')

    ax[0,1].plot(x_earth, y_earth)
    ax[0,1].set_xlabel('x (AU)')
    ax[0,1].set_ylabel('y (AU)')

    ax[0,2].plot(y_earth, z_earth)
    ax[0,2].set_xlabel('y (AU)')
    ax[0,2].set_ylabel('z (AU)')

    ax[1,0].plot(x_mars, z_mars)
    ax[1,0].set_xlabel('x (AU)')
    ax[1,0].set_ylabel('z (AU)')

    ax[1,1].plot(x_mars, y_mars)
    ax[1,1].set_xlabel('x (AU)')
    ax[1,1].set_ylabel('y (AU)')

    ax[1,2].plot(y_mars, z_mars)
    ax[1,2].set_xlabel('y (AU)')
    ax[1,2].set_ylabel('z (AU)')


    for b in ax:
        for a in b:
            a.grid()
            a.set_ylim(-2, 2)
            a.set_xlim(-2, 2)

    plt.tight_layout()
    plt.show()

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_earth.to_value(), y_earth.to_value(), z_earth.to_value(), cmap='Blues', c=time_spice)
    ax.scatter(x_mars.to_value(), y_mars.to_value(), z_mars.to_value(), cmap='Reds', c=time_spice)
    ax.scatter(x_sun.to_value(), y_sun.to_value(), z_sun.to_value(), color='orange')
    ax.set_xlabel('AU')
    ax.set_ylabel('AU')
    ax.set_zlabel('AU')

    ax.view_init(azim=-60, elev=10)

    plt.show()

    plot_full(101, save_plot=False, save_dir='plots\\')


def plot_full(i, save_plot=False, save_dir='solo_plots'):
    """
    Function to make a plot that contains several subplots including
    a 3D plot of the orbit of Solar Orbiter, Earth, Venus, and a plot
    of the orbits in x-y and x-z plane, and also include the speed and
    elevation of Solar Orbiter as a function of time.

    Parameters
    ----------
    i : `int`
        index of the time_spice to plot, i.e. it will plot
        positions at time step time_spice[i]

    save_plot : `Boolean`, optional
        if set True, then will save the plot to the `save_dir` directory

    save_dir : `str`, optional
        path to save the plot if save_plot==True.
        Defaults to creating a solo_plots directory in the
        current working directory

    """
    # setting up some style parameters that look good for this plot
    sns.set_context('paper', font_scale=0.8, rc={'axes.linewidth': 0.5})


    earth_col = '#377eb8'
    cmap_earth = sns.light_palette(earth_col, as_cmap=True)

    mars_col = 'k'
    cmap_mars = 'Greys'

    # i is for each timestep - but we also want to plot the
    # previous 20 timesteps, j, to illustrate the trajectory path
    if i < 20:
        j = 0
    else:
        j = i - 20

    # set up plotting informatio in each dictionary
    kwargs_Earth = {'s': 10, 'c': time_spice[j:i], 'cmap': cmap_earth}
    kwargs_Mars = {'s': 5, 'c': time_spice[j:i], 'cmap': cmap_mars}

    # get box sizes for plotting positions on figure
    xx = 10
    yy = 5
    box = 0.18

    fig = plt.figure(figsize=(xx, yy))

    # 3D plot of trajectories
    ax = plt.axes([0.0, 0.02, 0.6, 0.90], projection='3d')
    ax.scatter(x_earth.to_value()[j:i], y_earth.to_value()[j:i], z_earth.to_value()[j:i], **kwargs_Earth)
    ax.scatter(x_mars.to_value()[j:i], y_mars.to_value()[j:i], z_mars.to_value()[j:i], **kwargs_Mars)
    ax.scatter(x_sun.to_value()[i], y_sun.to_value()[i], z_sun.to_value()[i], color='y', s=30)

    ax.scatter(x_earth.to_value()[i], y_earth.to_value()[i], z_earth.to_value()[i], color=earth_col, label='Earth',
               s=10)
    ax.scatter(x_mars.to_value()[i], y_mars.to_value()[i], z_mars.to_value()[i], color=mars_col, label='Mars',
               s=5)

    ax.plot(x_earth.to_value()[0:i], y_earth.to_value()[0:i], z_earth.to_value()[0:i], color=earth_col, lw=0.2)
    ax.plot(x_mars.to_value()[0:i], y_mars.to_value()[0:i], z_mars.to_value()[0:i], color=mars_col, lw=0.1)

    ax.view_init(azim=-60, elev=20)

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-0.5, 0.5)
    ax.set_title('Mars and Earth Trajectory {:s}'.format(times[i].strftime('%Y-%m-%d %H:%M')), y=1.05, fontsize=12)
    ax.set_xlabel('x (AU)')
    ax.set_ylabel('y (AU)')
    ax.set_zlabel('z (AU)')
    leg = ax.legend(loc='upper right', bbox_to_anchor=(0.53, 0.85),
                    bbox_transform=plt.gcf().transFigure, fontsize=8)
    #for handle, text in zip(leg.legendHandles, leg.get_texts()):
    #    text.set_color(handle.get_facecolor()[0])

    # x-y plane projection plot
    bx = plt.axes([0.61, 0.08, box, box * (xx / yy)])
    bx.scatter(x_sun.to_value()[i], y_sun.to_value()[i], color='y', s=30)
    bx.scatter(x_earth.to_value()[j:i], y_earth.to_value()[j:i], **kwargs_Earth)
    bx.scatter(x_mars.to_value()[j:i], y_mars.to_value()[j:i], **kwargs_Mars)
    bx.tick_params(direction='in', labelleft=False, left=False, bottom=False, labelbottom=False, width=0.5,
                   length=3)

    bx.plot(x_earth.to_value()[0:i], y_earth.to_value()[0:i], color=earth_col, lw=0.2)
    bx.plot(x_mars.to_value()[0:i], y_mars.to_value()[0:i], color=mars_col, lw=0.1)
    bx.set_xlabel('x-y plane (AU)')

    bx.set_xlim(-1.5, 1.5)
    bx.set_ylim(-1.5, 1.5)
    bx.set_xticks([-1, -0.5, 0, 0.5, 1])
    bx.set_yticks([-1, -0.5, 0, 0.5, 1])

    # x-z plane projection plot
    cx = plt.axes([0.61 + box + 0.01, 0.08, box, box * (xx / yy)])
    cx.scatter(x_sun.to_value()[i], z_sun.to_value()[i], color='y', s=30)
    cx.scatter(x_earth.to_value()[j:i], z_earth.to_value()[j:i], **kwargs_Earth)
    cx.scatter(x_mars.to_value()[j:i], z_mars.to_value()[j:i], **kwargs_Mars)
    cx.plot(x_earth.to_value()[0:i], z_earth.to_value()[0:i], color=earth_col, lw=0.2)
    cx.plot(x_mars.to_value()[0:i], z_mars.to_value()[0:i], color=mars_col, lw=0.1)
    cx.tick_params(labelleft=False, labelbottom=False, left=False, right=False, bottom=False, labelright=False,
                   direction='in', width=0.5, length=3)
    cx.set_xlabel('x-z plane (AU)')

    cx.set_xlim(-1.05, 1.05)
    cx.set_ylim(-0.14, 0.25)
    cx.set_xticks([-1, -0.5, 0, 0.5, 1])
    cx.set_yticks([-0.1, 0, 0.1, 0.2])

    plt.show()
    # save each timestep plot
    if save_plot:
        # if save_dir is not provided then create a solo_plots path
        if save_dir == 'solo_plots':
            save_dir = os.path.join(os.getcwd(), save_dir)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        # this will save the files is ~/solo_plots/all_plots_0001.png for i = 1
        plt.savefig(os.path.join(save_dir, 'all_plots_{:04d}.png'.format(i)), dpi=250)
        plt.close()



if __name__ == '__main__':
    main()