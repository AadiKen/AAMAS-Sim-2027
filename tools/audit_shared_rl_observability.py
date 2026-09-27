"""Exact observation alias, with constant-velocity futures as an information test."""
import json,math
from pathlib import Path
from bcod_sim.benchmark.core import BenchmarkConfig,VesselReading,observation
config=BenchmarkConfig();rows=[]
for heading,speed in ((-math.pi/2,1),(math.pi/2,1),(0,0)):
    reading=VesselReading(0,0,0,1,0,nearby_agents=[(10,10)])
    obs=observation(reading,(30,0),config).tolist()
    future=(10+10*speed*math.cos(heading),10+10*speed*math.sin(heading))
    rows.append({'other_pose':[10,10,heading],'other_speed':speed,'observation':obs,
                 'separation_after_10s_constant_velocity':math.dist((10,0),future)})
assert all(r['observation']==rows[0]['observation'] for r in rows)
Path('artifacts/shared-rl-audit/observation-alias.json').write_text(json.dumps(rows,indent=2))
