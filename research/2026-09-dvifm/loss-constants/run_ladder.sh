#!/usr/bin/env bash
# §6/§7 experiment ladder. One `fit` run per arm emits artefact_<name>.json
# + ck_* checkpoints under report/fits/. Bins are v2 (mean-carrying) so both
# raw and Weber axes run on the same caches.
#   run_ladder.sh rehearse   — dev-leg rehearsal (cheap, proves the ladder)
#   run_ladder.sh humans     — cid22a + tidkadid + kadid + tid_jp2kjpeg arms
#   run_ladder.sh safesyn    — the headline arms (safesyn, weber, majority)
set -euo pipefail
L=/mnt/v/output/zensim/dvifm-loss-2026-09-20
C=$L/cache; P=$L/pairs; T=$L/tools
S=/mnt/v/output/zensim/dvifm-screen2d-2026-09-19
OUT=$L/report/fits
mkdir -p "$OUT"

# plane_bins <plane> dom1,dom2,...  ->  bin for <plane> in each domain,
# comma-joined in domain order (CatCache domain order == pairs flag order).
plane_bins () {
  local pl=$1 doms=$2 out=()
  IFS=',' read -ra D <<< "$doms"
  for d in "${D[@]}"; do out+=("$C/${d}_ycbcr_${pl}.bin"); done
  (IFS=','; echo "${out[*]}")
}

# dom_pairs dom1,dom2,... -> one --pairs flag per domain, same order.
dom_pairs () {
  local doms=$1 flag=${2:---pairs}
  IFS=',' read -ra D <<< "$doms"
  for d in "${D[@]}"; do echo -n "$flag $P/${d}.tsv "; done
}

# fit <name> <fit-doms> <dev-doms> [extra...]
# fit-doms: comma list of domain names; bins joined per plane, pairs in order.
# The heavy lock wraps EACH fit call (per-compute-step acquisition,
# released between arms) — never held across the whole ladder.
fit () {
  local name=$1 fit_doms=$2 dev_doms=$3; shift 3
  ~/tmp/devin/heavy -- python3 "$T/fit_loss.py" fit \
    --cache "ycbcr_y=$(plane_bins y "$fit_doms")" \
    --cache "ycbcr_cb=$(plane_bins cb "$fit_doms")" \
    --cache "ycbcr_cr=$(plane_bins cr "$fit_doms")" \
    $(dom_pairs "$fit_doms") \
    --dev-cache "ycbcr_y=$(plane_bins y "$dev_doms")" \
    --dev-cache "ycbcr_cb=$(plane_bins cb "$dev_doms")" \
    --dev-cache "ycbcr_cr=$(plane_bins cr "$dev_doms")" \
    --dev-pairs "$P/safesyn_development.tsv" \
    --init-spec "$S/specs/dvifm-ycbcr_y.json" --out-dir "$OUT" --name "$name" \
    --grids "$@" 2>&1 | tee "$OUT/log_$name.txt"
}

# like fit but dev legs come from a comma list of pairs TSVs (human domains
# have per-domain dev legs; safesyn's dev leg is named differently).
fit2 () {
  local name=$1 fit_doms=$2 dev_doms=$3 devpairs=$4; shift 4
  ~/tmp/devin/heavy -- python3 "$T/fit_loss.py" fit \
    --cache "ycbcr_y=$(plane_bins y "$fit_doms")" \
    --cache "ycbcr_cb=$(plane_bins cb "$fit_doms")" \
    --cache "ycbcr_cr=$(plane_bins cr "$fit_doms")" \
    $(dom_pairs "$fit_doms") \
    --dev-cache "ycbcr_y=$(plane_bins y "$dev_doms")" \
    --dev-cache "ycbcr_cb=$(plane_bins cb "$dev_doms")" \
    --dev-cache "ycbcr_cr=$(plane_bins cr "$dev_doms")" \
    $(dom_pairs "$devpairs" --dev-pairs) \
    --init-spec "$S/specs/dvifm-ycbcr_y.json" --out-dir "$OUT" --name "$name" \
    --grids "$@" 2>&1 | tee "$OUT/log_$name.txt"
}

case "${1:?usage: run_ladder.sh rehearse|humans|safesyn|safesyn_weber|majority}" in
  rehearse)
    fit safesyn_dev safesyn_dev safesyn_dev \
      --steps 60 --bootstrap 2 --boot-steps 30 --row-frac 0.5 ;;
  humans)
    for d in cid22a tidkadid kadid_train tid_jp2kjpeg; do
      fit2 "$d" "${d}_fit" "${d}_dev" "${d}_dev" \
        --steps 150 --bootstrap 20 --boot-steps 60
      fit2 "${d}_weber" "${d}_fit" "${d}_dev" "${d}_dev" \
        --steps 150 --bootstrap 0 --weber-eps 0.01
    done ;;
  safesyn)
    # headline: safesyn alone, raw axis
    fit safesyn safesyn_fit safesyn_dev \
      --steps 150 --bootstrap 20 --boot-steps 60 --row-frac 0.3
    # weber axis — the psychovisual-slope comparison
    fit safesyn_weber safesyn_fit safesyn_dev \
      --steps 150 --bootstrap 20 --boot-steps 60 --row-frac 0.3 \
      --weber-eps 0.01
    # safesyn-majority: 0.6/0.2/0.2 domain mass, raw axis.
    # dev legs = the three domains' dev splits, weights applied on dev too.
    fit2 majority safesyn_fit,cid22a_fit,tidkadid_fit \
      safesyn_dev,cid22a_dev,tidkadid_dev \
      safesyn_development,cid22a_dev,tidkadid_dev \
      --steps 150 --bootstrap 20 --boot-steps 60 --row-frac 0.3 \
      --domain-weights '{"0":0.6,"1":0.2,"2":0.2}' ;;
  safesyn_weber)
    fit safesyn_weber safesyn_fit safesyn_dev \
      --steps 150 --bootstrap 20 --boot-steps 60 --row-frac 0.3 \
      --weber-eps 0.01 ;;
  majority)
    fit2 majority safesyn_fit,cid22a_fit,tidkadid_fit \
      safesyn_dev,cid22a_dev,tidkadid_dev \
      safesyn_development,cid22a_dev,tidkadid_dev \
      --steps 150 --bootstrap 20 --boot-steps 60 --row-frac 0.3 \
      --domain-weights '{"0":0.6,"1":0.2,"2":0.2}' ;;
esac
