import datetime
import numpy as np
import spiceypy
from sunpy.time import TimeRange
from astropy import time, units as u
from astropy.coordinates import solar_system_ephemeris


def main():
    print("Running solar system model")

    spice_files = ['/Users/lahayes/spiceypy_test/kernels/lsk/naif0012.tls',
                   '/Users/lahayes/spiceypy_test/kernels/spk/de421.bsp',
                   '/Users/lahayes/spiceypy_test/kernels/pck/pck00010.tpc',
                   '/Users/lahayes/spiceypy_test/kernels/spk/solo_ANC_soc-orbit_20200207-20300902_V01.bsp']
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


if __name__ == '__main__':
    main()