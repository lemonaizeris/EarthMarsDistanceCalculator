import datetime
import numpy as np
import spiceypy
from sunpy.time import TimeRange
from astropy import units as u
from astropy.coordinates import solar_system_ephemeris
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

import seaborn as sns

import os
import subprocess


def main():
    print("Running solar system model...")

    spice_files = ['naif0012.tls',
                   'de441_part-1.bsp',
                   'de441_part-2.bsp',
                   'mar097s.bsp',
                   'mars_iau2000_v1.tpc',
                   'pck00011.tpc']
    spiceypy.furnsh(spice_files)

    timerange = TimeRange('2024-01-01', 10 * u.year)
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
    positions_mars, lightTimes_mars = spiceypy.spkezr('Mars', time_spice, 'ECLIPJ2000', 'NONE', 'Sun')
    positions_phobos, lightTimes_phobos = spiceypy.spkezr('Phobos', time_spice, 'ECLIPJ2000', 'NONE', 'Sun')
    positions_sun, lightTimes_sun = spiceypy.spkezr('Sun', time_spice, 'ECLIPJ2000', 'NONE', 'Sun')

    global per_object, x_object, y_object, z_object
    per_object = {}
    x_object = {}
    y_object = {}
    z_object = {}

    extract_object(positions_earth, 'earth')
    extract_object(positions_sun, 'sun')
    extract_object(positions_mars, 'mars')

    per_sun = np.array(positions_sun)[:, :3] * u.km
    global x_sun, y_sun, z_sun
    x_sun = per_sun[:, 0].to(u.au)
    y_sun = per_sun[:, 1].to(u.au)
    z_sun = per_sun[:, 2].to(u.au)

    do_calculations()

    # print('AFTER WHILE LOOP:')
    # print(dist_mars_earth)
    # print(len(dist_mars_earth))
    # print(len(times))
    # While loop is probably not so efficient. Maybe change to some other distance calculation method.
    # Commented code currently doesn't work

    plot_full(101, save_plot=False, save_dir='current_plots\\')
    #make_movie(0, len(x_object['earth'])-1, save_dir='current_plots\\')


def extract_object(object_position, object_name):
    per_object[object_name] = np.array(object_position)[:, :3] * u.km

    x_object[object_name] = per_object[object_name][:, 0].to(u.au)
    y_object[object_name] = per_object[object_name][:, 1].to(u.au)
    z_object[object_name] = per_object[object_name][:, 2].to(u.au)

def do_calculations():
    global dist_mars_earth, angle_mars_earth
    # dist_mars_earth = np.array(np.linalg.norm(np.array(positions_mars)[:, :3] - np.array(positions_earth)[:, :3], axis=1)) * u.au
    # dist_mars_earth = np.array([np.sqrt(np.sum(point_earth-per_mars[index])**2, axis=0) for point_earth, index in per_earth])
    # dist_mars_earth = np.sqrt(np.sum((per_earth[:]-per_mars)**2, axis=0)).to(u.au)
    dist_mars_earth = np.array([])
    angle_mars_earth = np.array([])
    pos_index = 0
    while pos_index < len(times):
        dist = np.sqrt((x_object['earth'].to_value()[pos_index] - x_object['mars'].to_value()[pos_index]) ** 2 +
                       + (y_object['earth'].to_value()[pos_index] - y_object['mars'].to_value()[pos_index]) ** 2 +
                       + (z_object['earth'].to_value()[pos_index] - z_object['mars'].to_value()[pos_index]) ** 2)

        earth_mars_vector = np.array([[x_object['mars'].to_value()[pos_index] - x_object['earth'].to_value()[pos_index]],
                                      [y_object['mars'].to_value()[pos_index] - y_object['earth'].to_value()[pos_index]],
                                      [z_object['mars'].to_value()[pos_index] - z_object['earth'].to_value()[pos_index]]])
        earth_sun_vector = np.array([[x_object['sun'].to_value()[pos_index] - x_object['earth'].to_value()[pos_index]],
                                     [y_object['sun'].to_value()[pos_index] - y_object['earth'].to_value()[pos_index]],
                                     [z_object['sun'].to_value()[pos_index] - z_object['earth'].to_value()[pos_index]]])
        # print('NOT normalized:')
        # print(earth_mars_vector)
        # print(earth_sun_vector)
        earth_mars_vector_norm = earth_mars_vector / np.linalg.norm(earth_mars_vector)
        earth_sun_vector_norm = earth_sun_vector / np.linalg.norm(earth_sun_vector)
        # print('Normalized:')
        # print(earth_mars_vector_norm)
        # print(earth_sun_vector_norm)

        angle = np.arccos(np.dot(np.squeeze(earth_mars_vector_norm), np.squeeze(earth_sun_vector_norm)))

        dist_mars_earth = np.append(dist_mars_earth, dist)
        angle_mars_earth = np.append(angle_mars_earth, angle)
        # print('A post_index ' + str(pos_index) + 'has elements: ' + str(len(dist_mars_earth)))
        pos_index = pos_index + 1

    angle_mars_earth = np.rad2deg(angle_mars_earth) % 360

def make_movie(start, end, save_dir, save_file='save_movie_mars_earth_orbits.mp4'):
    for i in range(start, end):
        print(i, 'out of ', len(x_object['earth']))
        plot_full(i, save_plot=True, save_dir=save_dir)

    # subprocess is used here to call bash
    subprocess.call(['ffmpeg', '-r', '30', '-f', 'image2', '-s', '1920x1080', '-i',
                     save_dir + '/all_plots_%04d.png', '-vcodec',
                     'libx264', '-crf', '25', '-pix_fmt', 'yuv420p', save_file])


def plot_full(i, save_plot=False, save_dir='current_plots'):
    # setting up some style parameters that look good for this plot
    sns.set_context('paper', font_scale=0.8, rc={'axes.linewidth': 0.5})

    earth_col = '#377eb8'
    cmap_earth = sns.light_palette(earth_col, as_cmap=True)

    mars_col = 'red'
    cmap_mars = sns.light_palette(mars_col, as_cmap=True)

    # i is for each timestep - but we also want to plot the
    # previous 20 timesteps, j, to illustrate the trajectory path
    if i < 20:
        j = 0
    else:
        j = i - 20

    # set up plotting information in each dictionary
    kwargs_Earth = {'s': 10, 'c': time_spice[j:i], 'cmap': cmap_earth}
    kwargs_Mars = {'s': 5, 'c': time_spice[j:i], 'cmap': cmap_mars}

    # get box sizes for plotting positions on figure
    xx = 10
    yy = 5
    box = 0.18

    fig = plt.figure(figsize=(xx, yy))

    # 3D plot of trajectories
    ax = plt.axes([0.0, 0.02, 0.6, 0.90], projection='3d')
    ax.scatter(x_object['earth'].to_value()[j:i], y_object['earth'].to_value()[j:i], z_object['earth'].to_value()[j:i], **kwargs_Earth)
    ax.scatter(x_object['mars'].to_value()[j:i], y_object['mars'].to_value()[j:i], z_object['mars'].to_value()[j:i], **kwargs_Mars)
    ax.scatter(x_object['sun'].to_value()[i], y_object['sun'].to_value()[i], z_object['sun'].to_value()[i], color='y', s=30)

    ax.scatter(x_object['earth'].to_value()[i], y_object['earth'].to_value()[i], z_object['earth'].to_value()[i], color=earth_col, label='Earth',
               s=10)
    ax.scatter(x_object['mars'].to_value()[i], y_object['mars'].to_value()[i], z_object['mars'].to_value()[i], color=mars_col, label='Mars',
               s=5)

    ax.plot(x_object['earth'].to_value()[0:i], y_object['earth'].to_value()[0:i], z_object['earth'].to_value()[0:i], color=earth_col, lw=0.2)
    ax.plot(x_object['mars'].to_value()[0:i], y_object['mars'].to_value()[0:i], z_object['mars'].to_value()[0:i], color=mars_col, lw=0.1)

    # print("CALCULATED: ")
    # print(np.sqrt((x_earth.to_value()[0:i] - x_mars.to_value()[0:i])**2 + (y_earth.to_value()[0:i] - y_mars.to_value()[0:i])**2 + (z_earth.to_value()[0:i] - z_mars.to_value()[0:i])**2))
    # print("dist_mars_earth: ")
    # print(dist_mars_earth[i])

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
    # for handle, text in zip(leg.legendHandles, leg.get_texts()):
    #    text.set_color(handle.get_facecolor()[0])

    # x-y plane projection plot
    bx = plt.axes([0.61, 0.08, box, box * (xx / yy)])
    bx.scatter(x_object['sun'].to_value()[i], y_object['sun'].to_value()[i], color='y', s=30)
    bx.scatter(x_object['earth'].to_value()[j:i], y_object['earth'].to_value()[j:i], **kwargs_Earth)
    bx.scatter(x_object['mars'].to_value()[j:i], y_object['mars'].to_value()[j:i], **kwargs_Mars)
    bx.tick_params(direction='in', labelleft=False, left=False, bottom=False, labelbottom=False, width=0.5,
                   length=3)

    bx.plot(x_object['earth'].to_value()[0:i], y_object['earth'].to_value()[0:i], color=earth_col, lw=0.2)
    bx.plot(x_object['mars'].to_value()[0:i], y_object['mars'].to_value()[0:i], color=mars_col, lw=0.1)
    bx.set_xlabel('x-y plane (AU)')

    bx.set_xlim(-1.5, 1.5)
    bx.set_ylim(-1.5, 1.5)
    bx.set_xticks([-1, -0.5, 0, 0.5, 1])
    bx.set_yticks([-1, -0.5, 0, 0.5, 1])

    # x-z plane projection plot
    cx = plt.axes([0.61 + box + 0.01, 0.08, box, box * (xx / yy)])
    cx.scatter(x_object['sun'].to_value()[i], z_object['sun'].to_value()[i], color='y', s=30)
    cx.scatter(x_object['earth'].to_value()[j:i], z_object['earth'].to_value()[j:i], **kwargs_Earth)
    cx.scatter(x_object['mars'].to_value()[j:i], z_object['mars'].to_value()[j:i], **kwargs_Mars)
    cx.plot(x_object['earth'].to_value()[0:i], z_object['earth'].to_value()[0:i], color=earth_col, lw=0.2)
    cx.plot(x_object['mars'].to_value()[0:i], z_object['mars'].to_value()[0:i], color=mars_col, lw=0.1)
    cx.tick_params(labelleft=False, labelbottom=False, left=False, right=False, bottom=False, labelright=False,
                   direction='in', width=0.5, length=3)
    cx.set_xlabel('x-z plane (AU)')

    cx.set_xlim(-1.5, 1.5)
    cx.set_ylim(-0.14, 0.25)
    cx.set_xticks([-1, -0.5, 0, 0.5, 1])
    cx.set_yticks([-0.1, 0, 0.1, 0.2])

    ex = plt.axes([0.67, 0.50, 0.3, 0.2])
    ex.plot(times, dist_mars_earth, color=mars_col, lw=0.5)
    ex.scatter(times[j:i], dist_mars_earth[j:i], **kwargs_Mars)
    ex.set_xlim(times[0], times[-1])
    ex.axvline(times[i], color='k', lw=0.5)
    ex.tick_params(direction='in', width=0.5, length=3)
    ex.set_xlabel('Time (UT)')
    ex.set_ylabel('Distance between Earth and Mars (AU)')

    fx = plt.axes([0.67, 0.78, 0.3, 0.2])
    fx.plot(times, angle_mars_earth, color=mars_col, lw=0.5)
    fx.scatter(times[j:i], angle_mars_earth[j:i], **kwargs_Mars)
    fx.set_xlim(times[0], times[-1])
    fx.axvline(times[i], color='k', lw=0.5)
    fx.tick_params(direction='in', width=0.5, length=3)
    fx.set_xlabel('Time (UT)')
    fx.set_ylabel('Angle between Earth/Mars and Earth/Sun (deg)')

    # save each timestep plot
    if save_plot:
        # if save_dir is not provided then create a solo_plots path
        if save_dir == 'current_plots':
            save_dir = os.path.join(os.getcwd(), save_dir)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        # this will save the files is ~/solo_plots/all_plots_0001.png for i = 1
        plt.savefig(os.path.join(save_dir, 'all_plots_{:04d}.png'.format(i)), dpi=250)
        plt.close()
    else:
        plt.show()


if __name__ == '__main__':
    main()
