"""
T1001_4_cover_ES_kimi.py
========================
Requirement: T1001_4_cover_ES kimi.txt (parameters reference T1001_1_ buy_ES.py)
Call ibAutotrade.py function doTrade0: if there is an open short
position of 1 contract of the CME future ES, cover (buy back) the contract.

- connects to IB first (port 7497 TWS paper, retry logic)
- checks open positions; proceeds only when exactly 1 ES contract is short
- limit_price = live delayed quote (AutoTrade.get_rt_data)
- cover = 1, buy/sell/short = 0
- doTrade0 AFL rule: cover only when a short position exists
  (also cancels any pending stop-loss/take-profit for the symbol)

Safety: run only against a paper account; this places real (paper) orders.
"""

import math
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent                 # IB\autotrade\test
_AUTOTRADE_DIR = _HERE.parent                           # IB\autotrade
if str(_AUTOTRADE_DIR) not in sys.path:
    sys.path.insert(0, str(_AUTOTRADE_DIR))

from ibAutotrade import AutoTrade, doTrade0  # noqa: E402


def live_price(auto_trade, contract):
    """Live (delayed) price: last, else ask, else bid, else bid/ask mid."""
    rt = auto_trade.get_rt_data(contract)
    print(f"live delayed quote: last={rt['last']} bid={rt['bid']} ask={rt['ask']}")
    for v in (rt["last"], rt["ask"], rt["bid"]):
        if v is not None and not math.isnan(v) and v > 0:
            return float(v)
    mid = (rt["bid"] + rt["ask"]) / 2
    if not math.isnan(mid) and mid > 0:
        return float(mid)
    raise RuntimeError("no valid delayed quote received for pricing")


def find_es_position(auto_trade):
    """Return (contract, size) of the open CME future ES position, else None."""
    for pos in auto_trade.ib.positions():
        c = pos.contract
        # positions() returns partially-qualified contracts: exchange is
        # usually blank for futures, so don't require it to be "CME"
        if (c.secType == "FUT" and c.symbol == "ES"
                and c.exchange in ("", "CME") and pos.position != 0):
            return c, pos.position
    return None


def main():
    at = AutoTrade(host="127.0.0.1", port=7497, client_id=1, auto_connect=False)
    if not at.ensure_connected():
        print("ERROR: cannot connect IB after retries")
        return 1

    found = find_es_position(at)
    if found is None:
        print("no open CME future ES position - nothing to cover")
        return 0
    contract, size = found
    if size != -1:
        print(f"ES position size is {size}, expected -1 (short 1) - aborting"
              + ("; use the sell script to close a long" if size > 0 else ""))
        return 1
    at.ib.qualifyContracts(contract)
    print(f"open position: {contract.localSymbol} size={size}")

    price = live_price(at, contract)
    print(f"pricing from live price = {price}")

    rs = doTrade0(
        auto_trade=at,                          # already connected above
        contract=contract,
        buy=0,
        sell=0,
        short=0,
        cover=1,                                # cover the open short position
        limit_price=price,                      # live delayed last price
        stop_loss_price=0.0,                    # unused on the cover-exit path
        profit_take_price=0.0,                  # unused on the cover-exit path
        num_contracts=1,
        ib_controller_tick_size=2500,           # ES tick = 2500/10000 = 0.25
        is_intra_day=True,
        is_mkt_order=False,
        max_num_tick_size4_buy_short_slippage=1,
        max_num_tick_size4_stp_lmt_order_slippage=2,
        symbol_type=1,                          # 1 = future
        port=7497,
        host="127.0.0.1",
        client_id=1,
        connect_retries=3,
        connect_sleep_sec=2,
    )
    print(f"doTrade0 returned: {rs} (1 = cover order placed, 0 = none)")

    if rs == 1:
        at.ib.sleep(3)
        print("state after cover:")
        for tr in at.ib.openTrades():
            o = tr.order
            if tr.contract.symbol == "ES":
                print(f"  working: orderId={o.orderId} {o.action} {o.totalQuantity} "
                      f"{o.orderType} status={tr.orderStatus.status}")
        for p in at.ib.positions():
            if p.contract.symbol == "ES":
                print(f"  ES position={p.position}")
    return 0 if rs == 1 else 1


if __name__ == "__main__":
    sys.exit(main())
