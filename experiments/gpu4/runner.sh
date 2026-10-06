#!/bin/bash
# sequential GPU job runner; reads gpu4/jobs.txt dynamically ("name cmd..."), stops at a line "END"
cd "/c/Users/chris/OneDrive/Documents/Uni/Kaggle Comps/Predicting airline satisfaction"
i=1
while true; do
  line=$(sed -n "${i}p" experiments/gpu4/jobs.txt)
  if [ -z "$line" ]; then sleep 20; continue; fi
  [ "$line" = "END" ] && break
  name=${line%% *}; cmd=${line#* }
  echo "$(date +%T) START $name" >> experiments/gpu4/runner.log
  eval "$cmd" > experiments/gpu4/$name.log 2>&1
  echo "$(date +%T) END $name rc=$? $(grep -h 'CV AUC' experiments/gpu4/$name.log)" >> experiments/gpu4/runner.log
  i=$((i+1))
done
