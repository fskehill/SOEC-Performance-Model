import numpy as np

"""
Renewable Energy Profile Integration.
"""

SECONDS_PER_DAY = 24 * 60 * 60

def cloud_attenuation(t, cloud_events):

    t_arr = np.atleast_1d(np.asarray(t, dtype=float))
    factor = np.ones_like(t_arr)

    for ev in cloud_events:
        start, dur, depth = ev['start'], ev['duration'], ev['depth']
        ramp = ev.get('ramp', 0.5)
        end = start + dur

        ramp_in = (t_arr >= start - ramp) & (t_arr < start)
        flat = (t_arr >= start) & (t_arr <= end)
        ramp_out = (t_arr > end) & (t_arr <= end + ramp)

        frac_in = (t_arr[ramp_in] - (start - ramp)) / ramp
        factor[ramp_in] *= 1 - depth * (0.5 - 0.5 * np.cos(np.pi * frac_in))

        factor[flat] *= (1 - depth)

        frac_out = (t_arr[ramp_out] - end) / ramp
        factor[ramp_out] *= 1 - depth * (0.5 + 0.5 * np.cos(np.pi * frac_out))

    return factor if factor.size > 1 else factor[0]

def solar_irradiance_profile(t, t_sunrise_h=6.0, t_sunset_h=18.0, G_peak=1000.0, cloud_events=None):

    t_h = np.mod(t, SECONDS_PER_DAY) / 3600.0
    daylight_frac = (t_h - t_sunrise_h) / (t_sunset_h - t_sunrise_h)

    G = G_peak * np.sin(np.pi * np.clip(daylight_frac, 0.0, 1.0))
    is_daylight = (t_h >= t_sunrise_h) & (t_h <= t_sunset_h)
    G_clear = np.where(is_daylight, G, 0.0)

    if cloud_events:
        G_clear = G_clear * cloud_attenuation(t, cloud_events)

    return G_clear

def irradiance_to_current_density(G, G_ref=1000.0, j_max=1.2, j_standby=0.05):

    j_solar = j_max * (G / G_ref)
    return np.maximum(j_solar, j_standby)

def renewable_j_profile(t, t_sunrise_h=6.0, t_sunset_h=18.0,
                        G_peak=1000.0, j_max=1.2, j_standby=0.05,
                        cloud_events=None, g_ref=1000.0):
    
    G = solar_irradiance_profile(t, t_sunrise_h, t_sunset_h, G_peak, cloud_events)
    return irradiance_to_current_density(G, g_ref, j_max, j_standby)

DUBLIN_SEASONS = {
    'summer': {
        'label': 'Dublin Summer Solstice (~21 Jun)',
        't_sunrise_h': 4.95,
        't_sunset_h': 21.95,
        'G_peak': 500.0,
    },
    'winter': {
        'label': 'Dublin Winter Solstice (~21 Dec)',
        't_sunrise_h': 8.63,
        't_sunset_h': 16.12,
        'G_peak': 125.0,
    },
}