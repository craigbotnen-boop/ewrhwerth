#!/usr/bin/env python3
import sys,json
from pathlib import Path
import reproduce
stage=sys.argv[1]
if stage=='p8': r=reproduce.p8_rank_chain(3000)
elif stage=='p9': r=reproduce.p9_local_map(1999)
elif stage=='p10': r=reproduce.p10_whitney([3,4,6,8])
elif stage=='p12': r=reproduce.p12_nonlinear()
elif stage=='p13': r=reproduce.p13_links()
elif stage=='p14': r=reproduce.p14_rank(8)
elif stage=='p17': r=reproduce.p17_exact_i2_equivalence()
else: raise SystemExit(f'unknown stage {stage}')
out=Path(__file__).resolve().parent/'results'/f'{stage}.json'
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(r,indent=2))
print(stage,'PASS candidate')
