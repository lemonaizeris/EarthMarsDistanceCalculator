import datetime
import numpy as np
import spiceypy
from sunpy.time import TimeRange
from astropy import time, units as u
from astropy.coordinates import solar_system_ephemeris
import matplotlib.pyplot as plt

from pathlib import Path


def main():
    print("Running solar system model")

    spice_files = [Path('/SPICE/kernels/lsk/naif0012.tls'),
                   Path('/SPICE/kernels/spk/de421.bsp'),
                   Path('/SPICE/kernels/pck/pck00011.tpc')]
    spiceypy.furnsh(spice_files)

    timerange = TimeRange('2020-02-10', 8 * u.year)
    times = [timerange.start.datetime]
    t = times[0]
    while t < timerange.end:
        t = t + datetime.timedelta(hours=24)
        times.append(t)

    time_spice = [spiceypy.str2et(t.strftime('%Y-%m-%d %H:%M')) for t in times]

    solar_system_ephemeris.set("jpl")

    positions_earth, lightTimes_earth = spiceypy.spkezr('Earth', time_spice, 'ECLIPJ2000', 'NONE', 'Sun')
    positions_mars, lightTimes_mars = spiceypy.spkezr('Mars', time_spice, 'ECLIPJ2000', 'NONE', 'Sun')

    per_earth = np.array(positions_earth)[:, :3] * u.km
    x_earth = per_earth[:, 0].to(u.au)
    y_earth = per_earth[:, 1].to(u.au)
    z_earth = per_earth[:, 2].to(u.au)

    per_mars = np.array(positions_mars)[:, :3] * u.km
    x_mars = per_mars[:, 0].to(u.au)
    y_mars = per_mars[:, 1].to(u.au)
    z_mars = per_mars[:, 2].to(u.au)

    fig, ax = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(15, 5))
    ax[0].plot(x_earth, z_earth)
    ax[0].set_xlabel('x (AU)')
    ax[0].set_ylabel('z (AU)')

    ax[1].plot(x_earth, y_earth)
    ax[1].set_xlabel('x (AU)')
    ax[1].set_ylabel('y (AU)')

    ax[2].plot(y_earth, z_earth)
    ax[2].set_xlabel('y (AU)')
    ax[2].set_ylabel('z (AU)')

    ax[3].plot(x_mars, z_mars)
    ax[3].set_xlabel('x (AU)')
    ax[3].set_ylabel('z (AU)')

    ax[4].plot(x_mars, y_mars)
    ax[4].set_xlabel('x (AU)')
    ax[4].set_ylabel('y (AU)')

    ax[5].plot(y_mars, z_mars)
    ax[5].set_xlabel('y (AU)')
    ax[5].set_ylabel('z (AU)')

    for a in ax:
        a.grid()
        a.set_ylim(-1, 1)
        a.set_xlim(-1, 1)

    plt.tight_layout()



if __name__ == '__main__':
    main()