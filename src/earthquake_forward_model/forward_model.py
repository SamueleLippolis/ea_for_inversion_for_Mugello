"""Deterministic translation of the forward-model kernel in ``pqu7v2.f``.

The search loop is deliberately excluded: :func:`evaluate_forward_model`
evaluates one earthquake-source model and returns its synthetic intensities and
misfit.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math


@dataclass(frozen=True)
class SourceModelParameters:
    latitude: float
    longitude: float
    strike: float
    rake: float
    dip: float
    depth: float
    length_plus: float
    length_minus: float
    mach_plus: float
    mach_minus: float
    s_velocity: float
    seismic_moment: float
    samples: int = 512
    sampling_interval: float = 0.05

    def __post_init__(self) -> None:
        if self.depth <= 0 or self.length_plus <= 0 or self.length_minus <= 0:
            raise ValueError("depth and fault lengths must be positive")
        if self.s_velocity <= 0 or self.seismic_moment <= 0:
            raise ValueError("S-wave velocity and seismic moment must be positive")
        if self.samples < 2 or self.sampling_interval <= 0:
            raise ValueError("samples must be >= 2 and sampling interval positive")
        if abs(self.mach_plus) >= 1 or abs(self.mach_minus) >= 1:
            raise ValueError("sub-shear Mach numbers must have absolute value < 1")


@dataclass(frozen=True)
class IntensityObservation:
    longitude: float
    latitude: float
    intensity: int


@dataclass(frozen=True)
class ForwardModelResult:
    parameters: SourceModelParameters
    observations: tuple[IntensityObservation, ...]
    distances_km: tuple[float, ...]
    kinematic_values: tuple[float, ...]
    predicted_intensities: tuple[int, ...]
    residual: float


def load_intensity_observations(path: str | Path) -> tuple[IntensityObservation, ...]:
    observations = []
    for line_number, line in enumerate(Path(path).read_text().splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split()
        if len(fields) < 3:
            raise ValueError(f"{path}:{line_number}: expected longitude latitude intensity")
        observations.append(IntensityObservation(float(fields[0]), float(fields[1]), int(fields[2])))
    if not observations:
        raise ValueError(f"no observations found in {path}")
    return tuple(observations)


def _calculate_geodetic_azimuth(phi1: float, lon1: float, phi2: float, lon2: float) -> float:
    """Ellipsoidal azimuth, following AZIM in the Fortran source."""
    aj, esqrd = 1.0067395, 0.006694318
    psdc1, psdc2 = math.tan(phi1), math.tan(phi2)
    difa = lon2 - lon1
    original_difa = difa
    if difa == 0.0:
        difa = 1e-7
    cotaz = math.sin(phi1) * (
        psdc2 / (aj * psdc1)
        + esqrd * math.sqrt(aj + psdc2 * psdc2) / math.sqrt(aj + psdc1 * psdc1)
        - math.cos(difa)
    ) / math.sin(difa)
    sinaz = 1.0 / math.sqrt(1.0 + cotaz * cotaz)
    if original_difa == 0.0:
        return 0.0 if phi1 <= phi2 else math.pi
    if difa < 0.0:
        sinaz = -sinaz
    cosaz = cotaz * sinaz
    azimuth = math.atan2(sinaz, cosaz)
    return azimuth % (2.0 * math.pi)


def _calculate_meridian_arc(latitude: float) -> float:
    return (6367654.5001 * latitude - 16107.0347 * math.sin(2 * latitude)
            + 16.9762 * math.sin(4 * latitude) - 0.0223 * math.sin(6 * latitude))


def _project_geographic_coordinates(latitude: float,
                                    relative_longitude: float) -> tuple[float, float]:
    """International Ellipsoid 1924 projection used by GIPI (metres)."""
    alpha = [relative_longitude ** n / n for n in range(1, 7)]
    sin1, cos1 = math.sin(latitude), math.cos(latitude)
    en = 6378388.0 / math.sqrt(1.0 - 0.0067226700223 * sin1 * sin1)
    chi = 0.9996 * cos1 * en
    cos2 = cos1 * cos1
    cos4, cos6 = cos2 * cos2, cos2 * cos2 * cos2
    north = 0.9996 * _calculate_meridian_arc(latitude) + chi * sin1 * (
        alpha[1] - (1 / 6 - cos2 - cos4 / 99) * alpha[3]
        + (1 / 120 - cos2 / 2 + cos4) * alpha[5]
    )
    east = chi * (alpha[0] - (0.5 - cos2 - cos4 / 296) * alpha[2]
                  + (1 / 24 - cos2 / 1.2 + cos4 / 1.017 + cos6 / 50) * alpha[4])
    return north, east


def _calculate_station_geometry(model: SourceModelParameters,
                                site: IntensityObservation) -> tuple[float, float]:
    d2r = math.pi / 180.0
    epicentre_lat, epicentre_lon = model.latitude * d2r, model.longitude * d2r
    site_lat, site_lon = site.latitude * d2r, site.longitude * d2r
    azimuth = _calculate_geodetic_azimuth(epicentre_lat, epicentre_lon, site_lat, site_lon)
    theta_degrees = ((azimuth - model.strike * d2r) % (2 * math.pi)) / d2r
    station_xy = _project_geographic_coordinates(site_lat, site_lon - epicentre_lon)
    epicentre_xy = _project_geographic_coordinates(epicentre_lat, 0.0)
    distance = math.hypot(station_xy[0] - epicentre_xy[0],
                          station_xy[1] - epicentre_xy[1]) / 1000.0
    return distance, theta_degrees


def _calculate_radiation_pattern(form: int, theta: float, azimuth: float,
                                 strike: float, slip: float, dip: float) -> float:
    a11 = math.cos(slip)*math.cos(strike) + math.sin(slip)*math.cos(dip)*math.sin(strike)
    a12 = math.cos(slip)*math.sin(strike) - math.sin(slip)*math.cos(dip)*math.cos(strike)
    a13 = -math.sin(slip)*math.sin(dip)
    a21 = -math.sin(strike)*math.sin(dip)
    a22 = math.cos(strike)*math.sin(dip)
    a23 = -math.cos(dip)
    xb = math.sin(theta)*math.cos(azimuth)
    yb = math.sin(theta)*math.sin(azimuth)
    zb = math.cos(theta)
    xc = a11*xb + a12*yb + a13*zb
    yc = a21*xb + a22*yb + a23*zb
    if form == 2:
        if math.sin(theta) != 0.0:
            return (2*xc*yc*math.cos(theta) - a13*yc - a23*xc) / math.sin(theta)
        return (a23*a11 + a21*a13)*math.cos(azimuth) + (a12*a23 + a13*a22)*math.sin(azimuth)
    if form == 3:
        return (a12*yc + a22*xc)*math.cos(azimuth) - (a11*yc + a21*xc)*math.sin(azimuth)
    raise ValueError("only SV (2) and SH (3) radiation are used")


def _calculate_fault_motion_series(
        model: SourceModelParameters, distance: float, theta_degrees: float,
        length: float, mach: float) -> list[tuple[float, tuple[float, ...]]]:
    depth, dt = model.depth / length, model.sampling_interval
    phi, dip, rake = map(math.radians, (model.strike, model.dip, model.rake))
    re0, theta0 = distance / length, math.radians(theta_degrees % 360.0)
    along, m = mach >= 0.0, abs(mach)
    r0sq = re0*re0 + depth*depth
    r0 = math.sqrt(max(0.0, r0sq))
    i0 = math.atan(re0 / depth)
    cothe0 = math.cos(theta0)
    if not along:
        cothe0 = -cothe0
    costh0 = math.sin(i0) * cothe0
    series = []
    for j in range(model.samples):
        time = j*dt + r0
        a = 1.0 - m*m
        bhalf = m*(m*r0*costh0 - time)
        discriminant = max(0.0, bhalf*bhalf - a*(-m*m*(r0sq - time*time)))
        source_position = (-bhalf - math.sqrt(discriminant)) / a
        if abs(source_position) > 1.0:
            continue
        r = math.sqrt(max(0.0, r0sq + source_position**2 - 2*r0*source_position*costh0))
        if r == 0.0:
            continue
        costh = max(-1.0, min(1.0, (r0*costh0 - source_position) / r))
        cosi = depth / r
        re = math.sqrt(max(0.0, re0**2 + source_position**2 - 2*re0*source_position*cothe0))
        sini = re / r
        takeoff = math.atan2(sini, cosi)
        cth = costh if along else -costh
        if abs(abs(cth) - abs(sini)) > 1.1920929e-7 and sini != 0.0:
            moving_theta = math.acos(max(-0.999999, min(0.999999, cth/sini)))
        else:
            moving_theta = math.pi if cth*sini < 0 else 0.0
        if theta0 > math.pi:
            moving_theta = 2*math.pi - moving_theta
        station_azimuth = phi + moving_theta
        sh = _calculate_radiation_pattern(3, math.pi-takeoff, station_azimuth,
                                          phi, rake, dip)
        sv = _calculate_radiation_pattern(2, math.pi-takeoff, station_azimuth,
                                          phi, rake, dip)
        cp, sp = math.cos(station_azimuth), math.sin(station_azimuth)
        directivity = 1.0 / (1.0 - m*costh)
        scale = directivity * (1.0/r) * m
        series.append((time, (sh*cp*scale, -sh*sp*scale,
                              -sv*sp*cosi*scale, -sv*cp*cosi*scale)))
    return series


def _calculate_peak_kinematic_value(model: SourceModelParameters, distance: float,
                                    theta: float) -> float:
    first = _calculate_fault_motion_series(
        model, distance, theta, model.length_plus, model.mach_plus)
    second = _calculate_fault_motion_series(
        model, distance, theta, model.length_minus, model.mach_minus)
    if not first or not second:
        return 0.0
    unit1, unit2 = model.length_plus/model.s_velocity, model.length_minus/model.s_velocity
    dt2 = model.sampling_interval * unit2
    k2, maximum = 0, 0.0
    for t1_raw, y1 in first:
        t1 = t1_raw * unit1
        while k2 + 1 < len(second) and second[k2][0]*unit2 - t1 < -dt2/2:
            k2 += 1
        # Fortran stops combining samples when the second rupture reaches its
        # end marker; it does not keep reusing the last nonzero sample.
        if second[k2][0]*unit2 - t1 < -dt2/2:
            break
        y2 = second[k2][1]
        east = (y1[0]+y1[2])/unit1 + (y2[0]+y2[2])/unit2
        north = (y1[1]+y1[3])/unit1 + (y2[1]+y2[3])/unit2
        maximum = max(maximum, math.hypot(east, north))
    return maximum


def _round_half_away_from_zero(value: float) -> int:
    return math.floor(value + 0.5) if value >= 0 else math.ceil(value - 0.5)


def _calculate_macroseismic_intensity(log_kf: float, log_moment: float,
                                      law: int) -> float:
    if law == 1:
        result = 9.241 + log_kf*3.358 + 8.04e-27*(10**log_moment)
    elif law == 2:
        result = (118.488 - log_kf*0.9456 + log_kf**2*0.8101 - log_moment*9.686
                  + 0.2575*log_kf*log_moment + 0.2152*log_moment**2)
        if -0.9456 + 0.2575*log_moment + log_kf*1.62 < 0:
            vertex = (0.9456 - 0.2575*log_moment) / 1.62
            result = (118.488 - vertex*0.9456 + vertex**2*0.8101 - log_moment*9.686
                      + 0.2575*vertex*log_moment + 0.2152*log_moment**2)
    else:
        raise ValueError("intensity law must be 1 or 2")
    return min(result, 11.0)


def evaluate_forward_model(
        model: SourceModelParameters,
        observations: tuple[IntensityObservation, ...], *,
        intensity_law: int = 2) -> ForwardModelResult:
    """Run one forward model and compute the Fortran sum-of-squares fitness."""
    geometry = [_calculate_station_geometry(model, site) for site in observations]
    values = [_calculate_peak_kinematic_value(model, distance, theta)
              for distance, theta in geometry]
    positive_far = [value for value, (distance, _) in zip(values, geometry)
                    if distance > 5.0 and value > 0.0]
    if not positive_far:
        raise ValueError("model produced no positive kinematic values beyond 5 km")
    log_moment = math.log10(model.seismic_moment)
    near_intensity = _round_half_away_from_zero(
        _calculate_macroseismic_intensity(
            math.log10(max(positive_far)), log_moment, intensity_law))
    strike = math.radians(model.strike)
    predicted = []
    for site, value in zip(observations, values):
        xt = (site.longitude-model.longitude) * (111.117*math.cos(model.longitude*math.pi/180))
        yt = (site.latitude-model.latitude) * 111.117
        xr = xt*math.cos(strike) - yt*math.sin(strike)
        yr = xt*math.sin(strike) + yt*math.cos(strike)
        in_near_field = (-model.length_minus-5 <= yr <= model.length_plus+5 and -5 <= xr <= 5)
        if in_near_field:
            synthetic = near_intensity
        elif value > 0.0:
            synthetic = _round_half_away_from_zero(
                _calculate_macroseismic_intensity(
                    math.log10(value), log_moment, intensity_law))
        else:
            synthetic = 1
        predicted.append(max(1, min(11, synthetic)))
    residual = float(sum((synthetic-site.intensity)**2
                         for synthetic, site in zip(predicted, observations)))
    return ForwardModelResult(model, observations, tuple(x[0] for x in geometry),
                              tuple(values), tuple(predicted), residual)
