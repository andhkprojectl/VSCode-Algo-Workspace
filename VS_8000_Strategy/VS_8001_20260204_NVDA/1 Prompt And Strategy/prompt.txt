Please provide python program for algo trade strategy

requirement:
1. filename: VS_8000_Strategy/VS_8001_20260204_NVDA/1 Prompt And Strategy/strategyIRB1000_V1.py
2. in py file, create a class strategyIRB1000_V1, strategy logic inside the class
3. class strategyIRB1000_V1 can be a parameter to all testing program VS_0003_test backtest.py, otherTest.py, monteCarloSimulation.py and walkforwardTest.py
4. class strategyIRB1000_V1 also can be parameter of 5 minute live trade program VS_9000_LiveTrade/5min/inCubation_intraday1.py (TBD)
5. strategyIRB1000_V1 backgroup information:
    - time frame: 5minute
    - period: from 09:30 to 16:00 regular NASDAQ trading hour US NY time zone
    - symbol: NVDA    
6. strategyIRB1000_V1 logic:
    - define below variables
    - H: high price, L: low price, O: open price, C: close price, V: volume
    - iRbhLRange = H - L; iRbUpperTh = H - iRbhLRange*0.45; iRbLowerTh = L + iRbhLRange*0.45;
    - iRbBullish = IIf(C > L AND O > L AND C < iRbLowerTh AND O < iRbLowerTh, 1, 0);
    - iRbBearish = IIf(C < H AND O < H AND C > iRbUpperTh AND O > iRbUpperTh, 1, 0);
    - ema10 = EMA of C, 10 bars. ema20 = EMA of C, 20 bars
    - regSlopema10 = linear regression slope of ema10 , 3 bars
    - find difference betwwen close price and ema20 and save in variable ema20DiffCAbsRound0
        - ema20DiffCAbsRound0 = (absolute value of ema20 - C)*100, round 0, i.e. remove all decimal place
    - divide ema20DiffCAbsRound0 into 5 rank (value 1- 5) and save to variable ema20DiffCAbsRank
        - ema20DiffCAbsRank = 1 if 0 <= ema20DiffCAbsRound0 < 10 
        - ema20DiffCAbsRank = 2 if 10 <= ema20DiffCAbsRound0 < 21
        - ema20DiffCAbsRank = 3 if 21 <= ema20DiffCAbsRound0 < 35
        - ema20DiffCAbsRank = 4 if 35 <= ema20DiffCAbsRound0 < 66
        - ema20DiffCAbsRank = 5 if 66 <= ema20DiffCAbsRound0
    - find number of bars of each rank from variable ema20DiffCAbsRank from last 100 bars, save in variable sumEma20DiffCAbsRank
        e.g. if last 100 bars of ema20DiffCAbsRank has 30 rank 1, sumEma20DiffCAbsRank for rank 1 is 30
    - find difference of next 5 bar close price and current bar close price, save in variable rt10
    - find number of bars of each rank that is up, save in variable sumEma20DiffCAbsUpRankper
        - i.e. sumEma20DiffCAbsUpRankper ma
        - e.g. if last 100 bars of ema20DiffCAbsRank has 30 rank 1, within 30 bars, 20 bars of rt10 > 0, ema20DiffCAbsUpRankper for rank 1 is 20
    - find percentage of rank between, save in variale ema20DiffCAbsUpRankper. 
            - e.g. sumEma20Rankper = sumEma20DiffCAbsUpRankper*100/sumEma20DiffCAbsRank
    - buy signal: 
        buy00 = iRbBullish
            AND regSlopema10 > Ref(regSlopema10, -1)
            AND pRtEma20Rankper > 50
            AND ema20DiffCAbsRank >= 3
            AND pRtEma20Rankper >= Ref(pRtEma20Rankper, -1) AND pRtEma20Rankper >= Ref(pRtEma20Rankper, -2)
		;
        buy signal is triggered when previous bar buy00 is true;
        if live trade, buy price is current close price. slippage = 0.03
        else if backtest, buy price is current open price        

    - sell signal:
        - 5 bars after buy bar
        if live trade, buy price is current close price. slippage = 0.03
        else if backtest, buy price is current open price                



    - short signal: 
        short00 = iRbBearish
            AND regSlopema10 < Ref(regSlopema10, -1)
            AND pRtEma20Rankper < 50
            AND ema20DiffCAbsRank >= 3
            AND pRtEma20Rankper <= Ref(pRtEma20Rankper, -1) AND pRtEma20Rankper <= Ref(pRtEma20Rankper, -2)
            ;		
        short signal is triggered when previous bar short00 is true;
        if live trade, short price is current close price. slippage = 0.03
        - else if backtest, short price is current open price        
        

    - cover signal:
        - 5 bars after short bar
        if live trade, buy price is current close price. slippage = 0.03
        else if backtest, buy price is current open price                



    stop loss handle:   
    - For stop loss of buy, stop limit order:
        - if live trade, Stop Price = current close price - previous bar of ATR(7)*2.5. stop limit price = Stop Price - 0.06
        - else if backtest, , Stop Price = current close price - previous bar of ATR(7)*2.5. stop limit price = Stop Price - 0.06
    - For stop loss of short, stop limit order:
        - if live trade, Stop Price = current close price + previous bar of ATR(7)*2.5. stop limit price = Stop Price + 0.06
        - else if backtest, , Stop Price = current close price + previous bar of ATR(7)*2.5. stop limit price = Stop Price + 0.06    


7. Positiion size of each buy / short trade: 110 NVDA stocks. Sell/cover all stocks for each trade