#!/bin/sh
# VN traffic + autonomous-driving infrastructure (Town02): expert data -> BC full / V2I-only -> evaluation ->
# PPO fine-tune -> evaluation. Every step runs in its own process, so each gets its own Traffic Manager port.
export PYTHONIOENCODING=utf-8
PY=venv/Scripts/python.exe
T="--town Town02"; E="--episodes 6 --steps 2000"
port=${COOP_TM_PORT:-8010}
run() { port=$((port + 1)); COOP_TM_PORT=$port "$@"; }

# models / data of the 134-input observation (before the infrastructure) are kept aside
for d in preTrained_models/ppo_coop/Town02 checkpoints/PPO_coop/Town02; do
  [ -d "$d" ] && [ "$(ls -A $d)" ] && mkdir -p "${d%/Town02}_noinfra" && mv "$d" "${d%/Town02}_noinfra/" && mkdir -p "$d"
done
[ -f data/bc_Town02.npz ] && mv data/bc_Town02.npz data/bc_Town02_noinfra.npz
[ -f data/bc_policy_Town02.pth ] && mv data/bc_policy_Town02.pth data/bc_policy_Town02_noinfra.pth

run $PY clean_world.py Town02
echo "== COLLECT" && run $PY -u bc.py collect $T --vn --infra --steps 40000 &&
echo "== BC full" && run $PY -u bc.py train $T &&
echo "== BC v2i-only (no shared LiDAR)" && run $PY -u bc.py train $T --noshare &&
echo "== EVAL expert" && run $PY -u evaluate.py $T --vn --infra $E --expert --name infra_expert &&
echo "== EVAL bc_full" && run $PY -u evaluate.py $T --vn --infra $E --model data/bc_policy_Town02.pth --name infra_bc_full &&
echo "== EVAL bc_v2i" && run $PY -u evaluate.py $T --vn --infra $E --no-share --model preTrained_models/ppo_coop_noshare/Town02/bc_policy.pth --name infra_bc_v2i &&
echo "== FINETUNE" && run $PY -X faulthandler -u continuous_driver.py --exp-name ppo $T --vn-traffic --infra --load-checkpoint True --total-timesteps 100000 --episode-length 3000 &&
echo "== EVAL ppo_full" && run $PY -u evaluate.py $T --vn --infra $E --name infra_ppo_full
echo "PIPELINE EXITED code=$?"
