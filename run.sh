#!/bin/bash
feature_file="data/feature_data/f_sens.300.pkl"
age_model_file="data/models/yuml.age.feature"

if [ ! -f "$feature_file" ]; then
  python process_data.py
fi
if [ ! -f "$age_model_file" ]; then
  python stack_age.py
fi
python stack_loca.py
python main.py
