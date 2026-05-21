"""Generate zensim's binary castleCSF LUT from cvvdp-gpu's vendored data.

Output format (little-endian):
  0      8      magic              b"ZACUMCSF"   (Z-acumen-CSF marker)
  8      4      schema_version     u32   = 1
  12     4      n_l_bkg            u32   = 32
  16     4      n_rho              u32   = 32
  20     4      n_channels         u32   = 3
  24     4      reserved           u32   = 0
  28     4      ge_sigma           f32   = 1.5 (the cvvdp ge_sigma at which the LUT was generated)
  32     4      sensitivity_corr_db f32  = -0.279742 (cvvdp's calibration scalar, baked into the table values)
  36     128    log_l_bkg_axis     [f32; 32] LE
  164    128    log_rho_axis       [f32; 32] LE
  292    12288  log_s              [f32; 3*32*32] LE (C-order: channel-major × l_bkg × rho)
  12580  4      crc32              u32 of bytes [0..12576]

Total: 12 584 bytes.
"""
import re
import struct
import zlib
from pathlib import Path

LUT_RS = Path("/home/lilith/work/zen/zenmetrics/crates/cvvdp-gpu/src/kernels/csf_lut/v0_5_4.rs")
OUT = Path("/home/lilith/work/zen/zensim--acumen-foundation/zensim/data/castle_csf_v0_5_4_cvvdp.lut")

MAGIC = b"ZACUMCSF"
SCHEMA = 1
N_L = 32
N_R = 32
N_CH = 3
GE_SIGMA = 1.5
SENSITIVITY_CORR_DB = -0.279742


def parse_const(text, name):
    m = re.search(rf"pub const {name}\s*:\s*\[f32;\s*\d+\]\s*=\s*\[(.*?)\];", text, re.DOTALL)
    if not m:
        raise ValueError(f"Could not find const {name}")
    nums = re.findall(r"[-]?\d+\.\d+e?[+-]?\d*", m.group(1))
    return [float(x) for x in nums]


def main():
    text = LUT_RS.read_text()
    log_l = parse_const(text, "LOG_L_BKG_AXIS")
    log_r = parse_const(text, "LOG_RHO_AXIS")
    log_s_a = parse_const(text, "LOG_S_O0_C1")
    log_s_rg = parse_const(text, "LOG_S_O0_C2")
    log_s_yv = parse_const(text, "LOG_S_O0_C3")

    assert len(log_l) == N_L, f"log_l_bkg has {len(log_l)} not {N_L}"
    assert len(log_r) == N_R
    assert len(log_s_a) == N_L * N_R
    assert len(log_s_rg) == N_L * N_R
    assert len(log_s_yv) == N_L * N_R

    body = bytearray()
    body += MAGIC
    body += struct.pack("<I", SCHEMA)
    body += struct.pack("<I", N_L)
    body += struct.pack("<I", N_R)
    body += struct.pack("<I", N_CH)
    body += struct.pack("<I", 0)  # reserved
    body += struct.pack("<f", GE_SIGMA)
    body += struct.pack("<f", SENSITIVITY_CORR_DB)
    for v in log_l:
        body += struct.pack("<f", v)
    for v in log_r:
        body += struct.pack("<f", v)
    # Channel-major: A, RG, YV
    for v in log_s_a:
        body += struct.pack("<f", v)
    for v in log_s_rg:
        body += struct.pack("<f", v)
    for v in log_s_yv:
        body += struct.pack("<f", v)
    crc = zlib.crc32(bytes(body)) & 0xFFFFFFFF
    body += struct.pack("<I", crc)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes(bytes(body))
    print(f"Wrote {OUT} ({len(body)} bytes)")
    print(f"  magic    = {MAGIC.decode()}")
    print(f"  schema   = {SCHEMA}")
    print(f"  shape    = {N_CH} channels × {N_L} L_bkg × {N_R} rho")
    print(f"  ge_sigma = {GE_SIGMA}")
    print(f"  s_corr   = {SENSITIVITY_CORR_DB:+.6f} dB")
    print(f"  crc32    = 0x{crc:08x}")
    print(f"  channels: A → RG → YV (24 bytes/sample × 3 channels)")


if __name__ == "__main__":
    main()
