# Requirement: Break Billing Form Tier Pricing Description

## Goal
Develop a Python program that reads the input Excel file `testTLOBillingForm\BillingPricingForm.xls`, interprets the `tierPricingDesc` column, extracts structured tier-pricing data, and writes it to an output Excel file.

## Program
- **File path:** `BIllingForm\breakBillingFormTierPricingDesc.py`

## Input
- **File:** `testTLOBillingForm\BillingPricingForm.xls`
- Read all columns from the input file.
- Focus on column `tierPricingDesc`. If its value is **not blank**, interpret and extract the value, then write to the output file.

## Interpretation of `tierPricingDesc`

1. **tierHead1**
   - All string content **before the 1st occurrence of `.`** is named `tierHead`.
   - Save to variable `tierHead1`.
   - All interpretation below applies only to the string **after the first `.`**.

2. **Split after `.` by `;`**
   - Each resulting substring is named `subString1`.

3. **For each `subString1`**, two possible formats:

   - **format1** — `0 <unit> no charge`
     - `0` is both the starting range and the ending range.
     - Save `0` to variables `rangeFrom` and `rangeTo`.
     - Save `<unit>` to variable `unit1`.

   - **format2** — further split by `:` into `subString2`.
     - Each `subString1` (format2) has **3** `subString2` parts:
       - **1st `subString2`** — `tier<N>`, where `N` is an integer.
         - Save to variable `tier`.
       - **2nd `subString2`** — has 2 formats:
         - **2a:** `<from> - <to> <Unit>`
           - `<from>` -> `rangeFrom`
           - `<to>` -> `rangeTo`
           - `<Unit>` -> `unit1`
         - **2b:** `> <from> <unit>`
           - `<from>` -> `rangeFrom`
           - `rangeTo` -> blank
           - `<unit>` -> `unit1`
       - **3rd `subString2`** — `<currency><amount>`
         - `currency` is one of {`HKD`, `THB`, `MOP`, `RMB`, `VND`, `INR`, `USD`} -> variable `currency`.
         - `amount` is a float number (4 decimal places). If greater than 1000, the amount may contain a `,` (thousands separator) -> variable `amount`.

## Example

```
tierPricingDesc = "Base on number of ASN. tier1:1-5 ASN: RMB217.62; tier2: 6-10 ASN: RMB435.24; tier3: 11-15 ASN: RMB652.86; tier4: 16-20 ASNs: RMB870.48; tier5: >20 ASN: RMB1,088.10; 0 ASN no charge"
```

## Output File

- **Behavior:** Each run of the program **overwrites** the existing output file content.
- **Location:** `testTLOBillingForm\BillingPricingForm_tierPricing.xls`
- **Format:**
  - Row 1 is the header row.
  - One data row per extracted tier.

| Column | Header (Row 1) | Content / Variable |
|--------|----------------|--------------------|
| A | `VendorCode` | Value of column D from input file |
| B | `BuyerCode` | Value of column F from input file |
| C | `Tier` | variable `tier` |
| D | `rangeFrom` | variable `rangeFrom` |
| E | `rangeTo` | variable `rangeTo` |
| F | `Unit` | variable `unit1` |
| G | `currency` | variable `currency` |
| H | `Amount` | variable `amount` |
| I | `TierHeader` | variable `tierHead1` |
