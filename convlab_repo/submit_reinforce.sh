#!/bin/bash
CUDA_VISIBLE_DEVICES=1,2	python run.py demo_pc.json rule_reinforce train
