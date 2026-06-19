Please update existing python program VS_8001_20260204_NVDA/2 Strategy/S8001_4_GenerateFromPromptQwen37Max.py

requirement:
please update function _calculate_rank_statistics
    - update sum_rank as array of series. There are 5 rank of ema20DiffCAbsRank (calculate from function _calculate_rank). for each rank ema20DiffCAbsRank, calcuate sum_rank from each rank, i.e. there are 5 series of sum_rank for each rank
    - Similar to sum_rank, update sum_up_rank. there are 5 series of sum_up_rank for each rank
    - update percentage. If return percentage depends of current bar ema20DiffCAbsRank value. For example, if current bar ema20DiffCAbsRank = 1, return rank 1 of sum_up_rank*100/sum_rank, if current bar ema20DiffCAbsRank = 2, return rank 2 of sum_up_rank*100/sum_rank, similar from rank 3 to 5

