#!/usr/bin/env python3
import sys,itertools

def decode11(s):
    s=s.strip()
    if len(s)!=11 or ord(s[0])-63!=11:
        raise RuntimeError(f"bad n=11 graph6: {s[:20]!r}")
    mask=0;pos=0
    for c in s[1:]:
        v=ord(c)-63
        for sh in range(5,-1,-1):
            if pos>=55: break
            if (v>>sh)&1: mask|=1<<pos
            pos+=1
    return mask

def encode12(mask):
    out=[chr(75)];pos=0
    for _ in range(11):
        v=0
        for _ in range(6):
            v=(v<<1)|((mask>>pos)&1);pos+=1
        out.append(chr(v+63))
    return ''.join(out)

if len(sys.argv)!=2:
    raise SystemExit("usage: augment_regular11_port.py PORT_DEGREE")
d=int(sys.argv[1])
if not 1<=d<=11: raise SystemExit("port degree must be 1..11")
subs=[]
for comb in itertools.combinations(range(11),d):
    x=0
    for i in comb: x |= 1<<(55+i)
    subs.append(x)
bases=0; emitted=0; buf=[]
for line in sys.stdin:
    if not line.strip(): continue
    base=decode11(line);bases+=1
    for ext in subs:
        buf.append(encode12(base|ext)+"\n");emitted+=1
        if len(buf)>=8192:
            sys.stdout.write(''.join(buf));buf.clear()
if buf: sys.stdout.write(''.join(buf))
print(f"AUGMENT bases={bases} d={d} subsets_per_base={len(subs)} emitted={emitted}",file=sys.stderr)
