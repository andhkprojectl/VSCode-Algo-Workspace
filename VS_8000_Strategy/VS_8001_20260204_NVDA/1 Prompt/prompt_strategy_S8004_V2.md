Please update existing python program VS_8001_20260204_NVDA/2 Strategy/S8004_GenerateFromPromptQwen37Max.py

requirement:
1. add 1 condition for buy and short signal: only allow buy or short between 09:30:00 and 14:55:00
2. add 1 condition for sell and cover signal: if there is open position of NVDA on 15:55:00. If position is from long, force sell all position. If position is from short, force cover all position

