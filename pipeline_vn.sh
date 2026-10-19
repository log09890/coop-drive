#!/bin/sh
# VN traffic in Town02: expert data (shared + ego-only obs) -> BC x2 -> fixed-seed evaluation -> PPO fine-tune -> evaluation
export PYTHONIOENCODING=utf-8
PY=venv/Scripts/python.exe
T="--town Town02"
E="--episodes 6 --steps 2000"
$PY clean_world.py Town02
if [ "$1" != "--eval-only" ]; then  # sh pipeline_vn.sh --eval-only: reuse data/ and the BC models
  echo "== COLLECT" && $PY -u bc.py collect $T --vn --steps 40000 &&
  echo "== BC share" && $PY -u bc.py train $T &&
  echo "== BC noshare" && $PY -u bc.py train $T --noshare || exit 1
fi
echo "== EVAL expert" && $PY -u evaluate.py $T --vn $E --expert --name expert &&
echo "== EVAL bc_share" && $PY -u evaluate.py $T --vn $E --model data/bc_policy_Town02.pth --name bc_share &&
echo "== EVAL bc_noshare" && $PY -u evaluate.py $T --vn $E --no-share --model preTrained_models/ppo_coop_noshare/Town02/bc_policy.pth --name bc_noshare &&
echo "== FINETUNE" && $PY -X faulthandler -u continuous_driver.py --exp-name ppo $T --vn-traffic --load-checkpoint True --total-timesteps 100000 --episode-length 3000 &&
echo "== EVAL ppo_share" && $PY -u evaluate.py $T --vn $E --name ppo_share
echo "PIPELINE EXITED code=$?"
