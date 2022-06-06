# shall be put in /data1/reinforcement_platform/rl_learner_platform/log

import os
import ast

with open('loss.txt', 'rb') as f:
    try:  # catch OSError in case of a one line file 
        f.seek(-2, os.SEEK_END)
        while f.read(1) != b'\n':
            f.seek(-2, os.SEEK_CUR)
    except OSError:
        f.seek(0)
    last_line = f.readline().decode()
#print(last_line)

loss_dict = ast.literal_eval(last_line)
loss_str = loss_dict["loss"]
loss_str = loss_str.translate({ord(c): None for c in "[],"})
#print(loss_str)
print(",".join(loss_str.split()))
