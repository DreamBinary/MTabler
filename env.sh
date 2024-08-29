#!/bin/bash


pip wheel vllm -w ./pkgs
pip wheel deepspeed -w ./pkgs
pip wheel accelerate -w ./pkgs
pip wheel open_clip_torch -w ./pkgs