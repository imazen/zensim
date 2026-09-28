#!/usr/bin/env python3
"""gmsd lane: COLD cost — process start + PNG decode (zen codecs) + colour
conversion + score — for GMSD and its peers through ONE binary
(`zenmetrics score --metric M --reference A --distorted B`), so every arm
pays the same process/decode path and differs only in the metric.

Fixtures: the speed owner's `test_pair` content (byte-identical generator,
ported from zensim-bench/benches/ssim2_speed_bar.rs) at 64..4096 square,
written as PNG by a stdlib writer. Arms are run round-robin in a shuffled
order per round (interleaved, so box drift is common-mode); `floor` is
`zenmetrics --version` (process start + exit, no decode, no metric).

Usage: cold_bench.py --bin ZENMETRICS --out DIR [--rounds 30] [--threads 1]
                     [--cpus 2] [--sizes 64,256,1024,2048,4096]
"""
import argparse, json, os, random, statistics, struct, subprocess, time, zlib


def png(path, w, h, rows):
    def chunk(t, d):
        return struct.pack('>I', len(d)) + t + d + struct.pack('>I', zlib.crc32(t + d) & 0xffffffff)
    raw = b''.join(b'\x00' + bytes(r) for r in rows)
    with open(path, 'wb') as f:
        f.write(b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0))
                + chunk(b'IDAT', zlib.compress(raw, 6)) + chunk(b'IEND', b''))


def test_pair(n):
    """Byte-identical to ssim2_speed_bar.rs::test_pair(n, n)."""
    src_rows, dst_rows = [], []
    for y in range(n):
        sr, dr = bytearray(), bytearray()
        for x in range(n):
            base = (x * 255) // n
            tex = ((x * 7 + y * 13) % 32) * 3
            edge = 40 if (y // 16) % 2 == 0 else 0
            px = [(base + tex) & 255, (base + edge) & 255, ((255 - base) + tex // 2) & 255]
            sr += bytes(px)
            d = [(v // 12) * 12 for v in px]
            if x < n // 2 and y < n // 2:
                d[0] = min(255, d[0] + 18)
            dr += bytes(d)
        src_rows.append(sr); dst_rows.append(dr)
    return src_rows, dst_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bin', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--rounds', type=int, default=30)
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--cpus', default='2')
    ap.add_argument('--sizes', default='64,256,1024,2048,4096')
    ap.add_argument('--metrics', default='gmsd,ssim2,butteraugli,zensim')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    sizes = [int(s) for s in a.sizes.split(',')]
    fx = {}
    for n in sizes:
        r, d = f'{a.out}/ref_{n}.png', f'{a.out}/dist_{n}.png'
        if not (os.path.exists(r) and os.path.exists(d)):
            sr, dr = test_pair(n)
            png(r, n, n, sr); png(d, n, n, dr)
        fx[n] = (r, d)
    env = dict(os.environ, RAYON_NUM_THREADS=str(a.threads))
    pre = ['taskset', '-c', a.cpus, 'nice', '-n19']
    arms = ['floor'] + a.metrics.split(',')
    rng = random.Random(20260922)
    times = {n: {m: [] for m in arms} for n in sizes}
    scores = {}
    for n in sizes:
        r, d = fx[n]
        for _ in range(2):  # warm the page cache / binary, discarded
            for m in arms[1:]:
                subprocess.run(pre + [a.bin, 'score', '--metric', m, '--reference', r, '--distorted', d],
                               env=env, capture_output=True, check=True)
        for _ in range(a.rounds):
            order = arms[:]; rng.shuffle(order)
            for m in order:
                cmd = [a.bin, '--version'] if m == 'floor' else \
                      [a.bin, 'score', '--metric', m, '--reference', r, '--distorted', d]
                t0 = time.perf_counter()
                p = subprocess.run(pre + cmd, env=env, capture_output=True, text=True, check=True)
                times[n][m].append((time.perf_counter() - t0) * 1e3)
                if m != 'floor':
                    scores[(n, m)] = p.stdout.strip()[-200:]
        print(n, {m: round(statistics.median(v), 2) for m, v in times[n].items()}, flush=True)
    out = dict(schema='gmsd-lane-cold-v1', threads=a.threads, cpus=a.cpus, rounds=a.rounds,
               sizes=sizes, bin=a.bin,
               median_ms={str(n): {m: statistics.median(v) for m, v in times[n].items()} for n in sizes},
               p10_ms={str(n): {m: sorted(v)[len(v) // 10] for m, v in times[n].items()} for n in sizes},
               p90_ms={str(n): {m: sorted(v)[(9 * len(v)) // 10] for m, v in times[n].items()} for n in sizes},
               raw_ms={str(n): times[n] for n in sizes},
               last_stdout={f'{n}:{m}': s for (n, m), s in scores.items()})
    json.dump(out, open(f'{a.out}/cold_{a.threads}t.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
