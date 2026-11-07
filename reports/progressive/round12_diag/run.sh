#!/bin/bash
# relaunch after reboot: bash /tmp/claude-0/-home-user-appmonitor-core/e496f878-8276-5793-a91f-1f6d0128e252/scratchpad/prog12/ev12/run.sh
E=/tmp/claude-0/-home-user-appmonitor-core/e496f878-8276-5793-a91f-1f6d0128e252/scratchpad/prog12/ev12
cd $E && exec setsid nohup flock -n $E/evq.lock /home/user/appmonitor-core/.venv/bin/python $E/evq.py "$@" >> $E/evq.out 2>&1 < /dev/null &
