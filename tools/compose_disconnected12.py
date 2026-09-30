#!/usr/bin/env python3
import sys, itertools, math
from pathlib import Path
from collections import Counter

EXPECTED_CONNECTED={2:1,3:2,4:6,5:21,6:112,7:853,8:11117,9:261080,10:11716571}
EXPECTED_TOTAL=12344252

def decode_mask(line, expected_n):
    s=line.strip()
    if not s or ord(s[0])-63 != expected_n:
        raise RuntimeError(f"bad graph6 for n={expected_n}: {s[:20]!r}")
    mask=0; pos=0
    vals=[ord(c)-63 for c in s[1:]]
    for val in vals:
        for shift in range(5,-1,-1):
            if pos >= expected_n*(expected_n-1)//2: break
            if (val>>shift)&1: mask |= 1<<pos
            pos += 1
    return mask

def embed(mask,n,off):
    if off==0:
        return mask
    out=0; pos=0
    for j in range(1,n):
        for i in range(j):
            if (mask>>pos)&1:
                gi,gj=off+i,off+j
                gp=gj*(gj-1)//2+gi
                out |= 1<<gp
            pos += 1
    return out

def encode12(mask):
    chars=[chr(12+63)]
    pos=0
    for _ in range(11):
        val=0
        for _ in range(6):
            val=(val<<1)|((mask>>pos)&1)
            pos += 1
        chars.append(chr(val+63))
    return ''.join(chars)

def partitions(n,maxp=None):
    if n==0:
        yield ()
        return
    if maxp is None: maxp=n
    for p in range(min(maxp,n),1,-1):
        for r in partitions(n-p,p):
            yield (p,)+r

def emit_buffer(buf):
    sys.stdout.write(''.join(buf))
    buf.clear()

def main(root):
    root=Path(root)
    catalogs={}
    for n in range(2,10):
        path=root/f"conn{n}.g6"
        masks=[decode_mask(line,n) for line in path.open()]
        if len(masks)!=EXPECTED_CONNECTED[n]:
            raise RuntimeError(f"connected count mismatch n={n}: {len(masks)} != {EXPECTED_CONNECTED[n]}")
        catalogs[n]=masks
        print(f"CATALOG n={n} count={len(masks)}",file=sys.stderr)

    pts=list(partitions(12,10))
    total_expected=0
    for pt in pts:
        cc=Counter(pt); ways=1
        for n,m in cc.items():
            ways*=math.comb(EXPECTED_CONNECTED[n]+m-1,m)
        total_expected+=ways
        print(f"PARTITION {pt} expected={ways}",file=sys.stderr)
    if total_expected!=EXPECTED_TOTAL:
        raise RuntimeError((total_expected,EXPECTED_TOTAL))

    generated=0; buf=[]
    # Dominant partition 10+2: stream n=10; bit positions of the n=10
    # component are unchanged at offset 0.  K2 on vertices 10,11 is bit 65.
    path10=root/"conn10.g6"
    c10=0
    for line in path10.open():
        m=decode_mask(line,10)
        buf.append(encode12(m | (1<<65))+"\n")
        c10+=1; generated+=1
        if len(buf)>=8192: emit_buffer(buf)
    if c10!=EXPECTED_CONNECTED[10]:
        raise RuntimeError(f"connected count mismatch n=10: {c10}")
    print(f"DONE partition=(10,2) generated={c10}",file=sys.stderr)

    # All remaining partitions have max part <= 9 and are small enough to
    # enumerate as products of connected-component catalogs.
    for pt in pts:
        if pt==(10,2): continue
        groups=[]
        offset=0
        # pt is nonincreasing, so equal-size components occupy consecutive offsets.
        for n,mult in Counter(pt).items():
            # Counter preserves first-occurrence order in pt.
            choices=itertools.combinations_with_replacement(range(len(catalogs[n])),mult)
            groups.append((n,mult,offset,choices))
            offset += n*mult
        part_count=0
        # Materialize choice iterables per group as needed. Groups with multiplicity
        # >1 occur only for small catalogs here.
        choice_lists=[]
        offset=0
        for n,mult in Counter(pt).items():
            if mult==1:
                choice_lists.append(((idx,) for idx in range(len(catalogs[n]))))
            else:
                choice_lists.append(itertools.combinations_with_replacement(range(len(catalogs[n])),mult))
        # itertools.product needs reusable iterables; convert only multi-group
        # smaller lists. Largest unique-size factor stays outermost.
        # Simpler recursive generator avoids materialization.
        items=list(Counter(pt).items())
        def rec(group_idx,off,mask):
            nonlocal generated,part_count
            if group_idx==len(items):
                buf.append(encode12(mask)+"\n"); generated+=1;part_count+=1
                if len(buf)>=8192: emit_buffer(buf)
                return
            n,mult=items[group_idx]
            cat=catalogs[n]
            if mult==1:
                for idx,m in enumerate(cat):
                    rec(group_idx+1,off+n,mask|embed(m,n,off))
            else:
                for combo in itertools.combinations_with_replacement(range(len(cat)),mult):
                    mm=mask; oo=off
                    for idx in combo:
                        mm |= embed(cat[idx],n,oo); oo += n
                    rec(group_idx+1,oo,mm)
        rec(0,0,0)
        print(f"DONE partition={pt} generated={part_count}",file=sys.stderr)

    if buf: emit_buffer(buf)
    print(f"COMPOSER generated={generated} expected={EXPECTED_TOTAL}",file=sys.stderr)
    if generated!=EXPECTED_TOTAL:
        raise RuntimeError(f"generated mismatch: {generated}")

if __name__=="__main__":
    if len(sys.argv)!=2:
        raise SystemExit("usage: compose_disconnected12.py CATALOG_DIR")
    main(sys.argv[1])
