"""
T1001_3_ short_ES_glm.py
========================
Requirement: T1001_3_ short_ES glm.txt (parameters reference T1001_1_ buy_ES.py)
Call VS_8101_autotrade.py function doTrade0 and short 1 contract of the CME
future ES.

- connects to IB first (port 7497 TWS paper, retry logic)
- prices from the live delayed quote (AutoTrade.get_rt_data)
- mirrored for a short position: stop_loss_price = price + 50,
  profit_take_price = price - 200
- doTrade0 AFL rule: short only when there is no open position

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

    price = live_price(at, contract)
    print(f"pricing from live price = {price}")

    rs = doTrade0(
        auto_trade=at,                          # already connected above
        contract=contract,
        buy=0,
        sell=0,
        short=1,                                # short 1 contract
        cover=0,
        limit_price=price,                      # live delayed last price
        stop_loss_price=price + 50,             # short: stop above entry
        profit_take_price=price - 200,          # short: target below entry
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
    print(f"doTrade0 returned: {rs} (1 = order placed, 0 = none)")

    if rs == 1:
        # wait until TWS acks the bracket children before disconnecting -
        # an immediate exit can cancel children still held (transmit=False)
        for _ in range(20):
            working = [tr for tr in at.ib.openTrades()
                       if tr.orderStatus.status in ("Submitted", "PreSubmitted")]
            if len(working) >= 2:
                break
            at.ib.sleep(0.5)
        at.ib.sleep(3)
        print("working orders:")
        for tr in at.ib.openTrades():
            o = tr.order
            print(f"  orderId={o.orderId} parentId={o.parentId} {o.action} "
                  f"{o.totalQuantity} {o.orderType} lmt={o.lmtPrice} "
                  f"stop={o.auxPrice} status={tr.orderStatus.status}")
        for p in at.ib.positions():
            if p.contract.symbol == "ES":
                print(f"  ES position={p.position}")
    return 0 if rs == 1 else 1


if __name__ == "__main__":
    sys.exit(main())
