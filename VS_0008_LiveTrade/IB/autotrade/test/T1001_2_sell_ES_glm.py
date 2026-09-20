"""
T1001_2_sell_ES_glm.py
======================
Requirement: T1001_2_sell_ES_glm.txt
If a CME future ES position is open (long), sell it via
VS_8101_autotrade.py function doTrade0 (sell-only exit order).

- connects to IB first (port 7497 TWS paper, retry logic)
- prices from the live delayed quote (AutoTrade.get_rt_data)
- doTrade0 sells the whole open ES position and cancels any pending
  stop-loss/take-profit orders for the symbol (AFL sell path)

Safety: run only against a paper account; this places real (paper) orders.
"""

import math
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent                 # IB\autotrade\test
_AUTOTRADE_DIR = _HERE.parent                           # IB\autotrade
if str(_AUTOTRADE_DIR) not in sys.path:
    sys.path.insert(0, str(_AUTOTRADE_DIR))

from ib_insync import Future  # noqa: E402

from VS_8101_autotrade import AutoTrade, doTrade0  # noqa: E402


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


def main():
    at = AutoTrade(host="127.0.0.1", port=7497, client_id=1, auto_connect=False)
    if not at.ensure_connected():
        print("ERROR: cannot connect IB after retries")
        return 1

    contract = Future("ES", "202612", "CME")
    at.ib.qualifyContracts(contract)

    position = at.get_position_size_with_retry("ES", sell=True, cover=False)
    print(f"current ES position = {position}")
    if position <= 0:
        print("no long ES position open - nothing to sell")
        return 0

    price = live_price(at, contract)
    print(f"pricing from live price = {price}")

    rs = doTrade0(
        auto_trade=at,                          # already connected above
        contract=contract,
        buy=0,
        sell=1,                                 # sell the open long position
        short=0,
        cover=0,
        limit_price=price,                      # current price (limit order)
        stop_loss_price=0.0,                    # exit order: no stop bracket
        profit_take_price=0.0,                  # exit order: no profit bracket
        num_contracts=1,
        ib_controller_tick_size=2500,           # ES tick = 2500/10000 = 0.25
        is_intra_day=True,
        is_mkt_order=False,
        max_num_tick_size4_buy_short_slippage=1,
        max_num_tick_size4_stp_lmt_order_slippage=0,
        symbol_type=1,                          # 1 = future
        port=7497,
        host="127.0.0.1",
        client_id=1,
        connect_retries=3,
        connect_sleep_sec=2,
    )
    print(f"doTrade0 returned: {rs} (1 = sell order placed, 0 = none)")

    if rs == 1:
        # hold briefly so TWS acks the sell order before disconnecting
        at.ib.sleep(3)
        print("state after sell:")
        for p in at.ib.positions():
            if p.contract.symbol == "ES":
                print(f"  ES position={p.position}")
        for tr in at.ib.openTrades():
            o = tr.order
            if tr.contract.symbol == "ES":
                print(f"  working: orderId={o.orderId} {o.action} {o.totalQuantity} "
                      f"{o.orderType} status={tr.orderStatus.status}")
    return 0 if rs == 1 else 1


if __name__ == "__main__":
    sys.exit(main())
