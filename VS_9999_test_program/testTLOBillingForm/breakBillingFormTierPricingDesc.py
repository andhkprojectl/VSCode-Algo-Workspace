import os
import re

import xlrd
import xlrd.book
import openpyxl

# The source workbook contains named formulas that xlrd cannot evaluate
# (FuncID:186). These named formulas are not needed to read cell data,
# so skip their evaluation to allow the workbook to open.
xlrd.book.evaluate_name_formula = lambda *a, **k: None

# Paths are kept relative per the requirement (testTLOBillingForm\...).
# They resolve against the directory that contains the BIllingForm folder.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE = os.path.join(BASE_DIR, "testTLOBillingForm", "BillingPricingForm.xls")
OUTPUT_FILE = os.path.join(BASE_DIR, "testTLOBillingForm", "BillingPricingForm_tierPricing.xls")

CURRENCIES = ("HKD", "THB", "MOP", "RMB", "VND", "INR", "USD")


def parse_amount(text):
    # <currency><amount>; amount is float, may contain ',' if > 1000, 4 dp.
    cur = ""
    for c in CURRENCIES:
        if text.strip().upper().startswith(c):
            cur = c
            break
    num_text = text.strip()[len(cur):].strip()
    amount = num_text.replace(",", "")
    return cur, amount


def parse_substring1(sub1):
    """Return a dict for one ';' segment, or None if it cannot be parsed."""
    sub1 = sub1.strip()
    if not sub1:
        return None

    # format1: 0 <unit> no charge
    m = re.match(r"0\s+(.+?)\s+no charge\s*$", sub1, re.IGNORECASE)
    if m:
        return {
            "tier": "0",
            "rangeFrom": "0",
            "rangeTo": "0",
            "unit1": m.group(1).strip(),
            "currency": "",
            "amount": "",
        }

    # format2: split by ':' into subString2 parts.
    parts = [p.strip() for p in sub1.split(":")]

    tier = ""
    rangeFrom = ""
    rangeTo = ""
    unit1 = ""
    currency = ""
    amount = ""

    # 1st subString2: tier<N>
    if len(parts) >= 1:
        tm = re.match(r"tier\s*(\d+)", parts[0], re.IGNORECASE)
        if tm:
            tier = tm.group(1)

    # 2nd subString2: range
    if len(parts) >= 2:
        rng = parts[1]
        if rng.startswith(">"):
            # format2b: > <from> <unit>
            rm = re.match(r">\s*([\d,]+)\s*(.*)$", rng)
            if rm:
                rangeFrom = rm.group(1).replace(",", "")
                rangeTo = ""
                unit1 = rm.group(2).strip()
        else:
            # format2a: <from> - <to> <Unit>
            rm = re.match(r"([\d,]+)\s*-\s*([\d,]+)\s*(.*)$", rng)
            if rm:
                rangeFrom = rm.group(1).replace(",", "")
                rangeTo = rm.group(2).replace(",", "")
                unit1 = rm.group(3).strip()

    # 3rd subString2: <currency><amount>
    if len(parts) >= 3:
        currency, amount = parse_amount(parts[2])

    return {
        "tier": tier,
        "rangeFrom": rangeFrom,
        "rangeTo": rangeTo,
        "unit1": unit1,
        "currency": currency,
        "amount": amount,
    }


def parse_tier_pricing_desc(desc):
    """Parse one tierPricingDesc value. Returns (tierHead1, list[dict])."""
    desc = (desc or "").strip()
    dot = desc.find(".")
    if dot != -1:
        tier_head1 = desc[:dot].strip()
        rest = desc[dot + 1:]
    else:
        tier_head1 = ""
        rest = desc

    rows = []
    for sub1 in rest.split(";"):
        parsed = parse_substring1(sub1)
        if parsed:
            rows.append(parsed)
    return tier_head1, rows


def main():
    book = xlrd.open_workbook(INPUT_FILE)
    sheet = book.sheet_by_index(0)

    # Column letters -> 0-based indices.
    col_vendor = 3   # D
    col_buyer = 5    # F
    col_tier = None
    header = sheet.row_values(0)
    for i, name in enumerate(header):
        if str(name).strip() == "tierPricingDesc":
            col_tier = i
            break
    if col_tier is None:
        raise ValueError("tierPricingDesc column not found in input file")

    out_wb = openpyxl.Workbook()
    out_ws = out_wb.active
    out_ws.title = "TierPricing"
    out_ws.append(
        ["VendorCode", "BuyerCode", "Tier", "rangeFrom", "rangeTo",
         "Unit", "currency", "Amount", "TierHeader"]
    )

    for r in range(1, sheet.nrows):
        vendor = sheet.cell_value(r, col_vendor)
        buyer = sheet.cell_value(r, col_buyer)
        desc = sheet.cell_value(r, col_tier)
        if str(desc).strip() == "":
            continue
        tier_head1, tier_rows = parse_tier_pricing_desc(desc)
        for row in tier_rows:
            out_ws.append([
                vendor, buyer, row["tier"], row["rangeFrom"], row["rangeTo"],
                row["unit1"], row["currency"], row["amount"], tier_head1,
            ])

    out_wb.save(OUTPUT_FILE)
    print(f"Wrote output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
