import sati
from datetime import datetime, timedelta
from skyfield.api import load, utc, EarthSatellite

def main():
    sat_list = load.tle('https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle')
    sat = sat_list['ISS (ZARYA)']
    bits = 0b11111111
    ts = load.timescale()

    t0 = datetime(2025, 8, 1, tzinfo=utc)
    t1 = t0 + timedelta(days=1)
    dt = timedelta(seconds=10)

    rec = sati.record(sat, bits, ts, t0, t1, dt, method="AACGM")
    tl = sati.timeline(rec, rec["MLT"] < 12)

if __name__ == "__main__":
    main()
