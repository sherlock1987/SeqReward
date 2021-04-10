#!/bin/bash
CUDA_VISIBLE_DEVICES=1,0	python run.py demo_pc.json rule_dqn train
