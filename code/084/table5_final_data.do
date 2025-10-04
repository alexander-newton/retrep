version 17
clear all
set more off

* ----- paths -----
global root "/Users/kimik/Desktop/RA (retransformation bias)/papers to be replicated/84/Replication_Files"
global data_dir "$root/data"

* ----- load raw -----
use "$data_dir/SCF_plus.dta", clear
mi unset, asis

* replicate paper tweaks
replace house  = house
replace ffaequ = ffaequ + mfun

* ---------- first collapse: thresholds (unweighted p90 for ffanw; weighted means) ----------
tempfile thresholds
preserve
collapse (p90) p90wealth = ffanw ///
         (mean) meanincome = tinc ///
         (mean) meanwealth = ffanw ///
         (mean) meanhouse  = house ///
         (mean) meanstocks = ffaequ [aw = wgtI95W95], by(yearmerge)
save `thresholds', replace
restore

* ---------- merge thresholds & keep top 10% ----------
merge m:1 yearmerge using `thresholds'
drop _merge
keep if ffanw > p90wealth

* ---------- second collapse over the top 10% ----------
collapse ///
    (mean) p90wealth = ffanw  ///
    (mean) p90income = tinc   ///
    (mean) meanincome = meanincome  ///
    (mean) meanwealth = meanwealth  ///
    (mean) house = house     ///
    (mean) stocks = ffaequ   ///
    (mean) meanhouse = meanhouse ///
    (mean) meanstocks = meanstocks [aw = wgtI95W95], by(yearmerge)

* ---------- shares ----------
gen top10wealthshare  = p90wealth/meanwealth * 0.1 
gen top10incomeshare  = p90income/meanincome * 0.1 
gen p90stockshare     = stocks/p90wealth
gen p90houseshare     = house/p90wealth
gen stockshare        = meanstocks/meanwealth
gen houseshare        = meanhouse/meanwealth

rename yearmerge year

* ---------- merge asset prices ----------
merge m:1 year using "$data_dir/assetprices.dta"
keep if _merge == 3
drop _merge

sort year
egen timeline = seq()
gen timeline2 = timeline*timeline
tsset timeline

* ---------- growths (match paper/Stata exactly) ----------
gen wealthgrowth = F.meanwealth/meanwealth
gen Y2W         = meanincome/meanwealth
gen p90Y2W      = p90income/p90wealth

gen stockgrowth = log(F.stockprice/stockprice)
label var stockgrowth "stock price growth"

gen housegrowth = log(F.houseprice/houseprice)
label var housegrowth "house price growth"

gen top10growth = log(F.top10wealthshare/top10wealthshare)
label var top10growth "growth top 10% wealth share"

* ---------- save processed dataset ----------
cap mkdir "$data_dir/processed"
save "$data_dir/processed/table5_baseline_prepared.dta", replace
display as text "Saved: $data_dir/processed/table5_baseline_prepared.dta"
