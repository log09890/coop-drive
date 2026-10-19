#!/bin/sh
# expert data (with braking + scenarios) -> behavior cloning -> PPO fine-tune; stops at the first failing stage
export PYTHONIOENCODING=utf-8
PY=venv/Scripts/python.exe
$PY clean_world.py
echo "== COLLECT" && $PY -u bc.py collect --steps 40000 &&
echo "== BC" && $PY -u bc.py train &&
echo "== FINETUNE" && $PY -X faulthandler -u continuous_driver.py --exp-name ppo --town Town07 --load-checkpoint True
echo "PIPELINE EXITED code=$?"
