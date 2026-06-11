please update python program VS_8001_20260204_NVDA/3 BackTest/backtest_S8004_V1.py
- There is error to function generate_explore_csv. please fix it because fucntion S8004_GenerateFromPromptQwen37Max.py is updated
- Also, please add column to function generate_explore_csv
    - data[f'sumEma20DiffCAbsRank_{r}']. r in range(1, 6). There are new 5 columns
    - data[f'sumEma20DiffCAbsUpRank_{r}']. r in range(1, 6). There are new 5 columns
    - data['pRtEma20Rankper']
