"""Reusable acceptance gates for CFD observations and convergence studies."""

from dataclasses import dataclass
import numpy as np


STATIONARITY_GATE_VERSION = "physical-scale-blend-v2"


@dataclass(frozen=True)
class QualityReport:
    accepted: bool
    residual_converged: bool
    stationary: bool
    finite: bool
    relative_drift: tuple[float, ...]
    reasons: tuple[str, ...]


def validate_run(wrench_history, residual_history, *, time_s, force_scale_n,
                 moment_scale_nm, window_s, residual_limit=1e-5) -> QualityReport:
    """Accept a CFD run only with residual and physically scaled load gates.

    Physical force/moment scales and actual sample times are mandatory; a
    percentage of a near-zero wrench channel is not a valid stationarity test.
    """
    wrench=np.asarray(wrench_history,float); residual=np.asarray(residual_history,float)
    finite=bool(wrench.ndim==2 and wrench.shape[1]==6 and len(wrench)>=5 and
                np.isfinite(wrench).all() and residual.size and np.isfinite(residual).all())
    if not finite: return QualityReport(False,False,False,False,(),("nonfinite or incomplete history",))
    stationarity=wrench_stationarity(time_s,wrench,force_scale_n=force_scale_n,
        moment_scale_nm=moment_scale_nm,window_s=window_s)
    stationary=stationarity.accepted; converged=bool(residual[-1]<=residual_limit)
    reasons=[]
    if not converged: reasons.append("residual threshold not met")
    if not stationary: reasons.append("force/moment history is not stationary")
    normalized_changes=tuple(channel.normalized_mean_change for channel in stationarity.channels)
    return QualityReport(converged and stationary,converged,stationary,True,normalized_changes,tuple(reasons))


def convergence_check(coarse, medium, fine, *, relative_limit=.05) -> dict:
    vectors=[np.asarray(x,float) for x in (coarse,medium,fine)]
    if any(x.shape!=(6,) or not np.isfinite(x).all() for x in vectors): raise ValueError("three finite 6-wrenches required")
    scale=np.maximum(np.abs(vectors[2]),1e-9)
    medium_fine=np.abs(vectors[1]-vectors[2])/scale
    coarse_medium=np.abs(vectors[0]-vectors[1])/np.maximum(np.abs(vectors[1]),1e-9)
    return {"accepted":bool(np.max(medium_fine)<=relative_limit),"relative_change_coarse_medium":coarse_medium.tolist(),
            "relative_change_medium_fine":medium_fine.tolist(),"limit":relative_limit}


def numerical_history_gate(diagnostics: dict, *, expected_start_s: float,
                           expected_end_s: float, max_courant: float,
                           max_alpha_courant: float, max_timestep_s: float,
                           residual_limit: float=1e-5,
                           max_relative_water_volume_error: float=.005) -> dict:
    """Audit complete saved OpenFOAM histories, including continuation coverage.

    A 0.001 absolute Courant allowance accommodates logged adaptive-step
    roundoff, not a changed physical or solver limit. Missing fields fail.
    """
    def finite(name):
        values=np.asarray(diagnostics.get(name,()),float)
        return values if values.size and np.isfinite(values).all() else np.asarray(())
    time=finite("time");courant=finite("courant_max");alpha=finite("alpha_courant_max")
    timestep=finite("timestep");volume=finite("water_volume_fraction")
    residuals=diagnostics.get("residuals",{})
    final_residuals=np.asarray([pair[1] for history in residuals.values() for pair in history],float)
    residual_valid=bool(final_residuals.size and np.isfinite(final_residuals).all())
    # PIMPLE records several pressure corrections per timestep. Test the last
    # corrector for each field/time, not an intermediate iteration. Raw
    # residuals remain available for diagnosis and are reported separately.
    step_finals=diagnostics.get("step_final_residuals",{})
    settled_residuals=(np.asarray([value for history in step_finals.values()
                                   for value in history],float) if step_finals else
                       np.asarray([pair[1] for history in residuals.values()
                           for pair in history[min(len(history)//100,10):]],float))
    volume_error=(float(np.max(np.abs(volume-volume[0]))/abs(volume[0]))
                  if volume.size and volume[0]!=0 else float("inf"))
    gates={
        "coverage":bool(time.size and np.all(np.diff(time)>0) and
            time[0]<=expected_start_s+2*max_timestep_s and
            time[-1]>=expected_end_s-2*max_timestep_s and diagnostics.get("solver_completed") is True),
        "max_courant":bool(courant.size and courant.max()<=max_courant+.001),
        "max_alpha_courant":bool(alpha.size and alpha.max()<=max_alpha_courant+.001),
        "timestep":bool(timestep.size and timestep.min()>0 and timestep.max()<=max_timestep_s*(1+1e-3)),
        "residual":bool(residual_valid and settled_residuals.size and
                        np.isfinite(settled_residuals).all() and settled_residuals.max()<=residual_limit),
        "water_volume":bool(volume_error<=max_relative_water_volume_error),
    }
    return {"accepted":all(gates.values()),"gates":gates,
        "observed":{"start_s":float(time[0]) if time.size else None,
                    "end_s":float(time[-1]) if time.size else None,
                    "max_courant":float(courant.max()) if courant.size else None,
                    "max_alpha_courant":float(alpha.max()) if alpha.size else None,
                    "max_timestep_s":float(timestep.max()) if timestep.size else None,
                    "max_final_residual":float(final_residuals.max()) if residual_valid else None,
                    "max_step_final_residual":float(settled_residuals.max()) if settled_residuals.size else None,
                    "intermediate_residual_exceedances":int(np.sum(final_residuals>residual_limit)) if residual_valid else None,
                    "max_relative_water_volume_error":volume_error}}


@dataclass(frozen=True)
class ChannelStationarity:
    regime: str
    mean: float
    standard_deviation: float
    minimum: float
    maximum: float
    slope_per_s: float
    first_half_mean: float
    second_half_mean: float
    normalized_mean_change: float
    normalized_slope: float
    normalized_standard_deviation: float
    accepted: bool
    periodic: "PeriodicStationarity | None" = None


@dataclass(frozen=True)
class PeriodicStationarity:
    accepted: bool
    frequency_hz: float
    period_s: float
    complete_cycles: int
    cycle_means: tuple[float, ...]
    cycle_amplitudes: tuple[float, ...]
    cycle_periods_s: tuple[float, ...]
    cycle_peak_to_peak: tuple[float, ...]
    normalized_mean_change: float
    normalized_mean_trend: float
    normalized_mean_variation: float
    relative_amplitude_change: float
    relative_amplitude_trend: float
    relative_amplitude_variation: float
    relative_period_change: float
    block_accepted: bool = False
    five_cycle_block_means: tuple[float, ...] = ()
    five_cycle_block_rms_amplitudes: tuple[float, ...] = ()


def periodic_stationarity(time_s, signal, *, physical_scale: float,
                          significant_scale_fraction=.001, min_cycles=5,
                          mean_change_limit=.03, mean_trend_limit=.03,
                          mean_variation_limit=.10, near_zero_mean_limit=.005,
                          near_zero_standard_deviation_limit=.01,
                          amplitude_change_limit=.10, amplitude_trend_limit=.10,
                          amplitude_variation_limit=.15, period_change_limit=.10
                          ) -> PeriodicStationarity | None:
    """Test cycle statistics, without demanding a flat instantaneous load.

    Returns None when there is no dominant periodic component or too few
    complete cycles. Detection uses a detrended spectrum, then narrowband
    peaks only to delimit cycles; cycle statistics use the unfiltered signal.
    """
    time=np.asarray(time_s,float); values=np.asarray(signal,float)
    if (time.ndim!=1 or values.shape!=time.shape or
        not np.isfinite(time).all() or not np.isfinite(values).all() or
        np.any(np.diff(time)<=0) or physical_scale<=0):
        raise ValueError("finite, increasing time and one signal are required")
    if len(time)<32: return None
    duration=float(time[-1]-time[0]); step=duration/4096
    uniform_t=np.linspace(time[0],time[-1],4097)
    uniform_y=np.interp(uniform_t,time,values)
    detrended=uniform_y-np.polyval(np.polyfit(uniform_t,uniform_y,1),uniform_t)
    frequency=np.fft.rfftfreq(len(uniform_t),step)
    spectrum=np.fft.rfft(detrended); amplitude=2*np.abs(spectrum)/len(uniform_t)
    candidates=(frequency>=min_cycles/duration)&(frequency<=min(5.,.45/step))
    if not np.any(candidates): return None
    indexes=np.flatnonzero(candidates); peak=int(indexes[np.argmax(amplitude[candidates])])
    dominant=float(frequency[peak])
    mean_magnitude=abs(float(uniform_y.mean()))
    # Do not promote a physically negligible cross-axis ripple to a periodic
    # gate just because its amplitude is large relative to a tiny mean.
    transition_fraction=max(significant_scale_fraction,near_zero_standard_deviation_limit)
    amplitude_weight=min(1.,mean_magnitude/(transition_fraction*physical_scale))
    material_amplitude=((1-amplitude_weight)*.005*physical_scale+
                        amplitude_weight*.05*mean_magnitude)
    if amplitude[peak] < max(.6*float(np.std(detrended)),material_amplitude): return None
    band=spectrum.copy();width=max(.15*dominant,1/duration)
    band[(frequency<dominant-width)|(frequency>dominant+width)]=0
    filtered=np.fft.irfft(band,n=len(uniform_t))
    peaks=np.flatnonzero((filtered[1:-1]>filtered[:-2])&
                           (filtered[1:-1]>=filtered[2:]))+1
    cycles=[]
    for start,stop in zip(peaks[:-1],peaks[1:]):
        segment_t=uniform_t[start:stop+1]; segment_y=uniform_y[start:stop+1]
        period=float(segment_t[-1]-segment_t[0])
        if not (.65/dominant <= period <= 1.35/dominant): continue
        phase=2*np.pi*(segment_t-segment_t[0])/period
        design=np.column_stack((np.ones(len(phase)),np.cos(phase),np.sin(phase)))
        coefficients=np.linalg.lstsq(design,segment_y,rcond=None)[0]
        cycles.append((float(segment_t[0]),period,
                       float(np.trapezoid(segment_y,segment_t)/period),
                       float(np.hypot(coefficients[1],coefficients[2])),
                       float(np.ptp(segment_y))))
    if len(cycles)<min_cycles: return None
    data=np.asarray(cycles); starts,periods,means,amps,peak_to_peak=data.T
    half=max(2,len(cycles)//2)
    mean_magnitude=abs(float(means.mean()))
    weight=min(1.,mean_magnitude/(transition_fraction*physical_scale))
    mean_change_denominator=((1-weight)*(near_zero_mean_limit/mean_change_limit)*physical_scale+
                             weight*mean_magnitude)
    other_mean_denominator=(1-weight)*physical_scale+weight*mean_magnitude
    mean_change=abs(float(means[-half:].mean()-means[:half].mean()))/mean_change_denominator
    mean_trend=abs(float(np.polyfit(starts,means,1)[0]))*duration/other_mean_denominator
    mean_variation=float(means.std())/other_mean_denominator
    amp_denominator=max(float(amps.mean()),np.finfo(float).eps)
    amp_change=abs(float(amps[-half:].mean()-amps[:half].mean()))/amp_denominator
    amp_trend=abs(float(np.polyfit(starts,amps,1)[0]))*duration/amp_denominator
    amp_variation=float(amps.std())/amp_denominator
    period_change=abs(float(periods[-half:].mean()-periods[:half].mean()))/float(periods.mean())
    accepted=(mean_change<=mean_change_limit and mean_trend<=mean_trend_limit and
              mean_variation<=mean_variation_limit and amp_change<=amplitude_change_limit and
              amp_trend<=amplitude_trend_limit and amp_variation<=amplitude_variation_limit and
              period_change<=period_change_limit)
    block_means=();block_rms=();block_accepted=False
    if len(cycles)>=15:
        # A steady superposition of free-surface modes can have a beating
        # envelope: one-cycle amplitudes alternate even though the statistics
        # of several complete cycles are stable. Use three non-overlapping
        # five-cycle blocks to test that longer-time mean and RMS envelope.
        recent=data[-15:];means5=[];rms5=[];periods5=[];centers5=[]
        for block in range(3):
            selected=recent[block*5:(block+1)*5]
            start=float(selected[0,0]);stop=float(selected[-1,0]+selected[-1,1])
            values5=uniform_y[(uniform_t>=start)&(uniform_t<=stop)]
            means5.append(float(values5.mean()));rms5.append(float(values5.std()))
            periods5.append(float(selected[:,1].mean()));centers5.append((start+stop)/2)
        block_means=tuple(means5);block_rms=tuple(rms5)
        block_magnitude=abs(float(np.mean(means5)))
        block_weight=min(1.,block_magnitude/(transition_fraction*physical_scale))
        block_change_den=((1-block_weight)*(near_zero_mean_limit/mean_change_limit)*physical_scale+
                          block_weight*block_magnitude)
        block_trend_den=(1-block_weight)*physical_scale+block_weight*block_magnitude
        span=centers5[-1]-centers5[0]
        mean_range=(max(means5)-min(means5))/block_change_den
        mean_slope=abs(float(np.polyfit(centers5,means5,1)[0]))*span/block_trend_den
        rms_den=max(float(np.mean(rms5)),np.finfo(float).eps)
        rms_range=(max(rms5)-min(rms5))/rms_den
        rms_slope=abs(float(np.polyfit(centers5,rms5,1)[0]))*span/rms_den
        period_range=(max(periods5)-min(periods5))/float(np.mean(periods5))
        block_accepted=(mean_range<=mean_change_limit and mean_slope<=mean_trend_limit and
                        rms_range<=amplitude_variation_limit and
                        rms_slope<=amplitude_trend_limit and
                        period_range<=period_change_limit)
    accepted=accepted or block_accepted
    return PeriodicStationarity(bool(accepted),dominant,1/dominant,len(cycles),
        tuple(map(float,means)),tuple(map(float,amps)),tuple(map(float,periods)),
        tuple(map(float,peak_to_peak)),float(mean_change),float(mean_trend),
        float(mean_variation),float(amp_change),float(amp_trend),
        float(amp_variation),float(period_change),bool(block_accepted),block_means,block_rms)


@dataclass(frozen=True)
class WrenchStationarity:
    accepted: bool
    channels: tuple[ChannelStationarity, ...]
    force_scale_n: float
    moment_scale_nm: float
    window_s: float


def wrench_stationarity(time_s, wrench, *, force_scale_n: float, moment_scale_nm: float,
                        window_s: float, significant_scale_fraction=.001,
                        mean_change_limit=.03, slope_limit=.03, standard_deviation_limit=.10,
                        near_zero_mean_limit=.005, near_zero_slope_limit=.005,
                        near_zero_standard_deviation_limit=.01) -> WrenchStationarity:
    """Scale-aware six-channel stationarity gate.

    Material channels retain mean-relative limits. A channel with mean below
    the existing near-zero standard-deviation allowance (1% of its declared
    physical scale by default) transitions continuously to the existing
    physical-scale absolute limits. For each metric j, the effective
    denominator is (1-w)*(absolute_limit_j/relative_limit_j)*scale
    + w*abs(mean), where w=min(1,abs(mean)/(transition_fraction*scale)).
    Thus no new tolerance is introduced: w=0 exactly reproduces the old
    near-zero absolute allowance and w=1 the old material-relative allowance.
    Slope is multiplied by the window duration before normalization.
    """
    time=np.asarray(time_s,float);values=np.asarray(wrench,float)
    if (time.ndim!=1 or values.shape!=(len(time),6) or len(time)<8 or
        not np.isfinite(time).all() or not np.isfinite(values).all() or
        np.any(np.diff(time)<=0) or force_scale_n<=0 or moment_scale_nm<=0 or window_s<=0):
        raise ValueError("finite time and [samples,6] wrench history required")
    end=float(time[-1]);mask=time>=end-window_s;window_t=time[mask];window=values[mask]
    if len(window)<8 or window_t[-1]-window_t[0] < .9*window_s: raise ValueError("stationarity window is incomplete")
    midpoint=end-window_s/2;first=window[window_t<midpoint];second=window[window_t>=midpoint]
    scales=(force_scale_n,)*3+(moment_scale_nm,)*3;channels=[]
    for axis,scale in enumerate(scales):
        y=window[:,axis];mean=float(y.mean());std=float(y.std());slope=float(np.polyfit(window_t,y,1)[0])
        first_mean=float(first[:,axis].mean());second_mean=float(second[:,axis].mean())
        transition_fraction=max(significant_scale_fraction,near_zero_standard_deviation_limit)
        weight=min(1.,abs(mean)/(transition_fraction*scale))
        mean_denominator=(1-weight)*(near_zero_mean_limit/mean_change_limit)*scale+weight*abs(mean)
        slope_denominator=(1-weight)*(near_zero_slope_limit/slope_limit)*scale+weight*abs(mean)
        std_denominator=(1-weight)*(near_zero_standard_deviation_limit/standard_deviation_limit)*scale+weight*abs(mean)
        regime="significant" if weight==1. else "near_zero"
        mean_change=abs(second_mean-first_mean)/mean_denominator
        normalized_slope=abs(slope)*window_s/slope_denominator
        normalized_std=std/std_denominator
        accepted=(mean_change<=mean_change_limit and normalized_slope<=slope_limit and
                  normalized_std<=standard_deviation_limit)
        periodic=periodic_stationarity(window_t,y,physical_scale=scale)
        if periodic is not None:
            # A detected but evolving oscillation cannot be hidden by stable
            # bulk statistics; conversely a stable periodic load need not be flat.
            accepted=periodic.accepted
            regime="periodic"
        channels.append(ChannelStationarity(regime,mean,std,float(y.min()),float(y.max()),slope,first_mean,
            second_mean,float(mean_change),float(normalized_slope),float(normalized_std),bool(accepted),periodic))
    return WrenchStationarity(all(x.accepted for x in channels),tuple(channels),force_scale_n,moment_scale_nm,window_s)
