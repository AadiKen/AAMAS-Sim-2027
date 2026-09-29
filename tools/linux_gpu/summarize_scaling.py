#!/usr/bin/env python3
"""Combine preserved scaling CSV segments without discarding partial counts."""
import argparse,csv,json,statistics
from collections import defaultdict
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('csv',nargs='+',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
rows=defaultdict(list)
for source in a.csv:
    with source.open(newline='') as stream:
        for row in csv.DictReader(stream):
            row['source_csv']=str(source)
            rows[int(row['count'])].append(row)
summary={'sources':[str(x) for x in a.csv],'counts':{},'notes':['Raw CSV files remain authoritative. Observed peak RSS is max(reported peak, end RSS); this corrects an early sampler ordering bug without inventing a measurement.','CLEAN_CANDIDATE means measured in a Slurm allocation with load telemetry; inspect host load before calling it uncontaminated.']}
for count,items in sorted(rows.items()):
    def med(key):return statistics.median(float(x[key]) for x in items)
    reps=[int(x['repetition']) for x in items]
    summary['counts'][str(count)]={'repetitions':len(items),'repetition_ids':reps,'complete_target_five':len(items)>=5,'statuses':sorted(set(x['status'] for x in items)),'median_aggregate_env_steps_per_s':med('aggregate_env_steps_per_s'),'median_vessel_steps_per_s':med('vessel_steps_per_s'),'median_p50_ms':med('p50_ms'),'median_p95_ms':med('p95_ms'),'median_p99_ms':med('p99_ms'),'max_observed_rss_bytes':max(max(int(x['rss_bytes']),int(x['peak_rss_bytes'])) for x in items),'median_cpu_percent_one_core':med('cpu_percent_one_core'),'max_load_average':max(float(x['load_average']) for x in items)}
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(summary,indent=2)+'\n');print(a.output)
