import datetime
import numpy as np
import spiceypy
from sunpy.time import TimeRange
from astropy import time, units as u
from astropy.coordinates import solar_system_ephemeris
import matplotlib.pyplot as plt
import jplephem

from pathlib import Path


def main():
    print("Running solar system model")

    spice_files = ['naif0012.tls',
                   'de430.bsp',
                   'mars_iau2000_v1.tpc',
                   'pck00011.tpc']
    spiceypy.furnsh(spice_files)

    timerange = TimeRange('2020-01-01', 1 * u.year)
    times = [timerange.start.datetime]
    t = times[0]
    while t < timerange.end:
        t = t + datetime.timedelta(hours=24)
        times.append(t)

    time_spice = [spiceypy.str2et(t.strftime('%Y-%m-%d %H:%M')) for t in times]

    solar_system_ephemeris.set("jpl")

    positions_earth, lightTimes_earth = spiceypy.spkezr('Earth', time_spice, 'ECLIPJ2000', 'NONE', 'Sun')
    positions_mars, lightTimes_mars = spiceypy.spkezr('Mars Barycenter', time_spice, 'ECLIPJ2000', 'NONE', 'Sun')
    positions_sun, lightTimes_earth = spiceypy.spkezr('Sun', time_spice, 'ECLIPJ2000', 'NONE', 'Sun')

    per_earth = np.array(positions_earth)[:, :3] * u.km
    x_earth = per_earth[:, 0].to(u.au)
    y_earth = per_earth[:, 1].to(u.au)
    z_earth = per_earth[:, 2].to(u.au)

    per_mars = np.array(positions_mars)[:, :3] * u.km
    x_mars = per_mars[:, 0].to(u.au)
    y_mars = per_mars[:, 1].to(u.au)
    z_mars = per_mars[:, 2].to(u.au)

    per_sun = np.array(positions_sun)[:, :3] * u.km
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



if __name__ == '__main__':
    main()