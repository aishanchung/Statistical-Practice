#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""2024→2025 5-VD space-time Kriging / multi-scale GP imputation benchmark.

Two correlated Gaussian-process models estimated using 2024 VD speeds only:
  1. st_kriging_ou: separable spatial covariance × exponential time kernel.
  2. gp_two_timescale: sum of fast and slow spatially correlated OU processes.

Both are *Gaussian processes*. For large time grids, the OU kernels admit
exact linear-Gaussian state-space / Kalman + RTS smoother inference. This is
NOT sklearn's dense O(N^3) GaussianProcessRegressor, and the chosen covariance
family is simple and stationary (not a full traffic-wave physics model).

No OD, weather, or V2 trip records. All imputation is retrospective and uses
observed times BEFORE AND AFTER each artificial/real gap.

Run Windows:
  py -m pip install duckdb numpy pandas scipy scikit-learn matplotlib
  py vd_gp_kriging_experiment.py VD_N3_N_2024_2025_LITE.parquet \
    --previous outputs/vd_spatial_methods_20261029 \
    --output vd_gp_kriging_results --blocks-per-stratum 2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from vd_imputation_experiment import VDExperiment, genuine_missing_runs
from vd_spatial_methods_benchmark import (aggregate, masked_patterns,
                                          class_confusion)

METHODS = ("st_kriging_ou", "gp_two_timescale")
PROBS = (.05, .10, .20, .30, .40, .50, .60, .70)


def positive_cov(x: np.ndarray, shrink: float = .18) -> np.ndarray:
    """Use fully observed training rows, spatial-shrink for PSD robustness."""
    clean = x[np.isfinite(x).all(axis=1)]
    if len(clean) < 5000:
        raise ValueError("Too few fully observed 2024 residual speed vectors")
    c = np.cov(clean, rowvar=False)
    out = (1-shrink)*c + shrink*np.diag(np.diag(c))
    return (out+out.T)/2 + .05*np.eye(x.shape[1])


def temporal_phi(x: np.ndarray, min_phi: float, max_phi: float) -> float:
    a, b = x[1:], x[:-1]
    okay = np.isfinite(a) & np.isfinite(b)
    first, second = a[okay], b[okay]
    estimate = float(np.dot(first, second) / (np.dot(second, second)+1.e-10))
    return float(np.clip(estimate, min_phi, max_phi))


class SpaceTimeGP:
    def __init__(self, components: list[tuple[float, np.ndarray]],
                 observation_variance: float = 9.0):
        """Each (phi, spatial covariance) defines OU temporal GP component."""
        self.comp = components
        self.ncomponents = len(components)
        self.d = 5 * self.ncomponents
        self.a = np.concatenate([np.full(5, phi) for phi, _ in components])
        self.h = np.concatenate([np.eye(5)] * self.ncomponents, axis=1)
        self.r = np.eye(5)*observation_variance
        self.p_stationary = np.zeros((self.d,self.d))
        self.q = np.zeros((self.d,self.d))
        for c, (phi, covariance) in enumerate(components):
            sl = slice(5*c,5*(c+1))
            self.p_stationary[sl,sl] = covariance
            self.q[sl,sl] = (1-phi*phi)*covariance
        self.identity = np.eye(self.d)

    def smooth(self, centered_observations: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
        """Pooled Kalman filtering and full offline RTS smoothing.

        Inputs shape [T,5] with NaN missing; outputs speed-residual posterior
        means and *new-observation* standard deviations, both [T,5].
        """
        observations = centered_observations
        n = len(observations)
        d = self.d
        filtered_mean = np.zeros((n,d))
        filtered_cov = np.zeros((n,d,d))
        predicted_mean = np.zeros((n,d))
        predicted_cov = np.zeros((n,d,d))
        state = np.zeros(d)
        covariance = self.p_stationary.copy()
        for t in range(n):
            if t:
                state = self.a * state
                covariance = (self.a[:,None]*covariance)*self.a[None,:] + self.q
            predicted_mean[t] = state
            predicted_cov[t] = covariance
            valid = np.flatnonzero(np.isfinite(observations[t]))
            if len(valid):
                h = self.h[valid]
                innovations = observations[t,valid] - h@state
                cross = covariance@h.T
                innovation_cov = h@cross + self.r[np.ix_(valid,valid)]
                gain = np.linalg.solve(innovation_cov, cross.T).T
                state = state + gain@innovations
                covariance = covariance - gain@cross.T
                covariance = (covariance+covariance.T)*.5
            filtered_mean[t] = state
            filtered_cov[t] = covariance
        # RTS backward: no hidden truth, but can use observed values AFTER gap.
        state = filtered_mean[-1].copy()
        covariance = filtered_cov[-1].copy()
        mean = np.empty((n,5))
        sd = np.empty((n,5))
        for t in range(n-1,-1,-1):
            if t < n-1:
                x = filtered_cov[t]*self.a[None,:]
                gain = np.linalg.solve(predicted_cov[t+1], x.T).T
                state = filtered_mean[t] + gain@(state-predicted_mean[t+1])
                covariance = filtered_cov[t] + gain@(covariance-predicted_cov[t+1])@gain.T
                covariance = (covariance+covariance.T)*.5
            mean[t] = self.h@state
            hc = self.h@covariance
            # Predict held-out observations (not noise-free latent speeds).
            predvar = np.sum(hc*self.h, axis=1)+np.diag(self.r)
            sd[t] = np.sqrt(np.maximum(predvar, 1.e-6))
        return mean, sd


class GPExperiment:
    def __init__(self, exp: VDExperiment, observation_noise_sd: float = 3.):
        self.e = exp
        resid = exp.v-exp.seasonal
        train = resid[exp.train].copy()
        # Single separable covariance: empirically estimated from 2024.
        covariance = positive_cov(train)
        lag = np.array([temporal_phi(train[:,i], .5, .99)
                        for i in range(5)])
        phi_single = float(np.median(lag))
        self.models = {
            "st_kriging_ou": SpaceTimeGP(
                [(phi_single, covariance)], observation_noise_sd**2),
        }
        # Two-component GP: within-hours shocks vs slowly evolving state.
        slow = pd.DataFrame(train).rolling(36, min_periods=12,
                                            center=True).mean().to_numpy()
        fast = train-slow
        c_slow = positive_cov(slow, shrink=.2)
        c_fast = positive_cov(fast, shrink=.2)
        phi_slow = float(np.median([temporal_phi(slow[:,i], .90, .997)
                                    for i in range(5)]))
        phi_fast = float(np.median([temporal_phi(fast[:,i], .25, .985)
                                    for i in range(5)]))
        self.models["gp_two_timescale"] = SpaceTimeGP(
            [(phi_fast,c_fast),(phi_slow,c_slow)], observation_noise_sd**2)
        self.meta = {
            "single_phi_5min": phi_single,
            "fast_phi_5min": phi_fast,
            "slow_phi_5min": phi_slow,
            "single_station_variances": np.diag(covariance).tolist(),
            "fast_station_variances": np.diag(c_fast).tolist(),
            "slow_station_variances": np.diag(c_slow).tolist(),
            "obs_noise_sd_kmh": observation_noise_sd,
        }
        print("GP fitted on 2024 VD-only residuals", self.meta, flush=True)

    def predict(self, start: int, end: int, masked_stations: tuple[int,...],
                pad_bins: int = 288):
        e=self.e
        lo,hi = max(0,start-pad_bins),min(e.n,end+pad_bins)
        source = e.v[lo:hi].copy()
        source[start-lo:end-lo,list(masked_stations)] = np.nan
        residual = source-e.seasonal[lo:hi]
        outputs={}
        for name, model in self.models.items():
            z,sd = model.smooth(residual)
            mean = e.seasonal[start:end]+z[start-lo:end-lo]
            outputs[name] = (np.clip(mean, 0, 150), sd[start-lo:end-lo])
        return outputs


def block_metrics(truth: np.ndarray, predicted: np.ndarray, sd: np.ndarray,
                  block_info: dict, station: str, masked: tuple[int,...],
                  method: str) -> dict:
    okay = np.isfinite(truth) & np.isfinite(predicted) & np.isfinite(sd)
    y = truth[okay]; p = predicted[okay]; s = sd[okay]
    err = p-y
    cm = class_confusion(y,p) if len(y) else np.zeros((3,3),int)
    z = norm.cdf((40-p)/s)
    l,u = p-1.645*s,p+1.645*s
    inside=(y>=l)&(y<=u)
    jam=(y<40)
    row = {
        "block_id": int(block_info["block_id"]),
        "start_time": str(block_info["start_time"]),
        "stratum": str(block_info["stratum"]),
        "duration_hours": float(block_info["duration_hours"]),
        "n_stations_missing": len(masked),
        "masked_indices": ",".join(map(str,masked)),
        "VDID": station, "method": method,
        "n_truth": int(np.isfinite(truth).sum()),
        "n_pred": len(y),
        "ae_sum": float(np.abs(err).sum()),
        "se_sum": float(np.square(err).sum()),
        "bias_sum": float(err.sum()),
        "interval90_hits": int(inside.sum()),
        "interval90_jam_hits": int((inside&jam).sum()),
        "interval90_jam_total": int(jam.sum()),
        "interval90_width_sum": float((u-l).sum()),
        "brier_p40_sum": float(np.square(z-(y<40)).sum()),
    }
    for a in range(3):
        for b in range(3):
            row[f"cm_{a}{b}"] = int(cm[a,b])
    for threshold in PROBS:
        predjam = z >= threshold
        truthjam = y<40
        tag = f"p{int(threshold*100):02d}"
        row[f"tp_{tag}"] = int((predjam&truthjam).sum())
        row[f"fp_{tag}"] = int((predjam&~truthjam).sum())
        row[f"fn_{tag}"] = int((~predjam&truthjam).sum())
    return row


def aggregate_uncertainty(detail: pd.DataFrame, keys: list[str]):
    base = aggregate(detail,keys)
    extra = detail.groupby(keys,as_index=False)[
        ["interval90_hits","interval90_jam_hits","interval90_jam_total",
         "interval90_width_sum","brier_p40_sum"]
    ].sum()
    out=base.merge(extra,on=keys)
    out["interval90_empirical_coverage"] = out.interval90_hits/out.n_pred.replace(0,np.nan)
    out["interval90_coverage_when_congested"] = (
        out.interval90_jam_hits/out.interval90_jam_total.replace(0,np.nan))
    out["interval90_mean_width_kmh"] = out.interval90_width_sum/out.n_pred.replace(0,np.nan)
    out["brier_p40"] = out.brier_p40_sum/out.n_pred.replace(0,np.nan)
    return out.drop(columns=["interval90_hits","interval90_jam_hits",
                             "interval90_jam_total","interval90_width_sum",
                             "brier_p40_sum"])


def probability_curves(detail: pd.DataFrame):
    keys=["stratum","duration_hours","n_stations_missing","method"]
    rec=[]
    for tags,group in detail.groupby(keys,sort=True):
        for threshold in PROBS:
            tag=f"p{int(threshold*100):02d}"
            tp,fp,fn=[int(group[f"{z}_{tag}"].sum()) for z in ("tp","fp","fn")]
            rec.append({**dict(zip(keys,tags)), "p_congestion_threshold":threshold,
                        "tp":tp,"fp":fp,"fn":fn,
                        "precision":tp/(tp+fp) if tp+fp else np.nan,
                        "recall":tp/(tp+fn) if tp+fn else np.nan,
                        "f1":2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else np.nan})
    return pd.DataFrame(rec)


def run_benchmark(e: VDExperiment, gp: GPExperiment, previous: Path,
                  output: Path, blocks_per_stratum: int, pad_bins: int,
                  hours_filter: list[float] | None = None):
    blocks=pd.read_csv(previous/"masked_blocks_manifest.csv")
    if hours_filter:
        blocks=blocks[blocks.duration_hours.isin(hours_filter)].copy()
    use=blocks.groupby(["duration_hours","stratum"],sort=False).head(
        blocks_per_stratum).copy()
    use.to_csv(output/"matched_blocks_manifest.csv",index=False,encoding="utf-8-sig")
    records=[]
    for n,(_,block) in enumerate(use.iterrows(),1):
        start,end=int(block.start_idx),int(block.end_idx)
        for mask in masked_patterns():
            pred=gp.predict(start,end,mask,pad_bins)
            for j in mask:
                truth=e.v[start:end,j]
                for name,(means,sds) in pred.items():
                    records.append(block_metrics(truth,means[:,j],sds[:,j],
                                                 block.to_dict(),e.ids[j],mask,name))
        print(f"GP validation block {n}/{len(use)}: "
              f"{block.duration_hours}h {block.stratum} {block.start_time}",flush=True)
    details=pd.DataFrame(records)
    details.to_csv(output/"gp_kriging_by_block_station.csv",index=False,
                   encoding="utf-8-sig")
    keys=["stratum","duration_hours","n_stations_missing","method"]
    summary=aggregate_uncertainty(details,keys)
    summary.to_csv(output/"gp_kriging_summary.csv",index=False,encoding="utf-8-sig")
    probability_curves(details).to_csv(output/"gp_probability_thresholds.csv",
                                       index=False,encoding="utf-8-sig")
    # Matched comparison: same held-out times, masks and VD, existing 9 methods.
    old=pd.read_csv(previous/"benchmark_by_block_station.csv")
    old=old[old.block_id.isin(use.block_id)].copy()
    combined=pd.concat([old,details[old.columns]],ignore_index=True)
    compare=aggregate(combined,keys)
    compare.to_csv(output/"matched_all_11_methods.csv",index=False,
                   encoding="utf-8-sig")
    return details,summary,compare


def run_real_missing(e:VDExperiment,gp:GPExperiment,output:Path,pad_bins:int):
    """Smooth contiguous periods with at least one originally missing VD.

    This is an optional, more expensive step. Never overwrites original panel.
    """
    rows=[]
    any_missing=e.missing.any(axis=1)
    runs=list(genuine_missing_runs(any_missing))
    for i,(start,end) in enumerate(runs,1):
        # Treat naturally absent values as missing, but retain observed stations.
        lo,hi=max(0,start-pad_bins),min(e.n,end+pad_bins)
        resid=e.v[lo:hi]-e.seasonal[lo:hi]
        preds={}
        for method,model in gp.models.items():
            z,sd=model.smooth(resid)
            preds[method]=(np.clip(e.seasonal[start:end]+z[start-lo:end-lo],0,150),
                           sd[start-lo:end-lo])
        for t in range(start,end):
            for j in np.flatnonzero(e.missing[t]):
                row={"TargetTime":str(e.times[t]),"VDID":e.ids[j],
                     "VDType_original":str(e.types[t,j]),
                     "Was_missing_original":True,
                     "Speed_original_code":float(e.raw[t,j]),
                     "other_VD_valid_speed_count":int(np.isfinite(e.v[t]).sum())}
                for name,(mean,sd) in preds.items():
                    mu=float(mean[t-start,j]); sigma=float(sd[t-start,j])
                    row[name+"_speed"] = mu
                    row[name+"_sd"] = sigma
                    row[name+"_lower90"] = mu-1.645*sigma
                    row[name+"_upper90"] = mu+1.645*sigma
                    row[name+"_p_congestion_lt40"] = float(norm.cdf((40-mu)/sigma))
                rows.append(row)
        if i%50==0 or i==len(runs):
            print(f"GP real missing runs {i}/{len(runs)}",flush=True)
    df=pd.DataFrame(rows).sort_values(["TargetTime","VDID"])
    assert len(df)==int(e.missing.sum())
    df.to_csv(output/"real_missing_gp_candidates.csv",index=False,
              encoding="utf-8-sig")
    return df


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("parquet",type=Path)
    p.add_argument("--previous",type=Path,
                   default=Path("outputs/vd_spatial_methods_20261029"))
    p.add_argument("--output",type=Path,default=Path("vd_gp_kriging_results"))
    p.add_argument("--blocks-per-stratum",type=int,default=2)
    p.add_argument("--context-hours",type=float,default=24.)
    p.add_argument("--hours",type=float,nargs="*",default=None,
                   help="Optional duration filter, e.g. --hours 6 72 151.5")
    p.add_argument("--obs-noise-sd",type=float,default=3.)
    p.add_argument("--skip-actual",action="store_true")
    p.add_argument("--only-actual",action="store_true",
                   help="Compute original-missing candidates only; keep benchmark outputs")
    a=p.parse_args()
    if a.blocks_per_stratum<1 or a.context_hours<0 or a.obs_noise_sd<=0:
        p.error("Invalid positive block count, context, or noise SD")
    if not a.parquet.exists():p.error(f"Missing {a.parquet}")
    if not (a.previous/"benchmark_by_block_station.csv").exists():
        p.error("--previous must contain previous full benchmark outputs")
    a.output.mkdir(parents=True,exist_ok=True)
    exp=VDExperiment(a.parquet)
    gp=GPExperiment(exp,a.obs_noise_sd)
    pad=int(round(a.context_hours*12))
    if a.only_actual:
        actual=run_real_missing(exp,gp,a.output,pad)
        print("Original missing candidates (not truth):",len(actual),flush=True)
        return
    details,summary,compare=run_benchmark(exp,gp,a.previous,a.output,
                                          a.blocks_per_stratum,pad,a.hours)
    if not a.skip_actual:
        actual=run_real_missing(exp,gp,a.output,pad)
        print("Original missing candidates (not truth):",len(actual),flush=True)
    previous_config_path=a.previous/"experiment_config.json"
    previous_config=(json.loads(previous_config_path.read_text(encoding="utf-8"))
                     if previous_config_path.exists() else {})
    metadata={"data":a.parquet.name,"train":2024,"mask_test":2025,
              "seed":previous_config.get("seed"),
              "block_subset":"First N blocks in each duration × stratum from previous benchmark; not independently redrawn",
              "blocks_per_stratum":a.blocks_per_stratum,
              "context_hours_each_side":a.context_hours,
              "missing_patterns":"contiguous 1..5 sensors",
              "gp_models":gp.meta,
              "model_equivalence":"Kriging and GP both Gaussian conditional covariance models; two different OU kernels compared",
              "interval":"Normal predictive 90% interval (untruncated), empirical coverage must be checked",
              "classification":"congested <40, heavy 40.. <80, clear >=80",
              "probability":"P(speed<40) from predictive normal; thresholds 0.05..0.70 are diagnostic only (not chosen on 2025)",
              "offline":"RTS smoothing uses observed future speeds after each masked gap",
              "long_gap_caution":"With all five VDs absent, GP reverts toward historical seasonal mean with high posterior uncertainty",
              "original":"Source Parquet is unchanged; Speed=-1 no vehicle is not congestion",
              "OD_weather":"not used",
             }
    (a.output/"experiment_config.json").write_text(
        json.dumps(metadata,ensure_ascii=False,indent=2),encoding="utf-8")
    print("\nSelected: 72h congestion-enriched: ")
    print(compare.query('stratum=="congestion_enriched" and duration_hours==72')
          [["n_stations_missing","method","coverage","mae_kmh",
            "congestion_recall","congestion_precision","congestion_f1"]]
          .round(3).to_string(index=False))
    print("Output:",a.output.resolve())


if __name__=="__main__":main()
