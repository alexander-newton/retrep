/*==============================================
TABLE 2
==============================================*/

* Set the data directory
global datadir "/Users/kimik/Desktop/RA (retransformation bias)/papers to be replicated/129/Datafortheweb/data"  //need to be changed to rawdata in our directory

/*==============================================
STEP 1: CONSTRUCT DATA == FORMATION OF NATIVES
==============================================*/

* use 50 year intervals

use "$datadir/reformation_towns.dta", clear

drop if cityname_cantoni == ""
drop if markettown==""
drop if cityid==.
drop if u1800 == .

* start with cross-section
merge 1:m cityid using "$datadir/db_data_city_decade_final.dta", gen(_city)

* collapse 
gen people = b_people
collapse (sum) people , by(cityid decade protestant_at_t ind_law u1800 cityname polity1500  river hansa  longitude latitude u1500 mean_plague1500_1521 mean_plague1400_1499 markettownid)

* assume "maturity" occurs at age 40
gen mature = decade + 40
gen period=1 if mature>=1220 &mature<1270
replace period=2 if mature>=1270 & mature<1320
replace period=3 if mature>=1320 & mature<1370
replace period=4 if mature>=1370 & mature<1420
replace period=5 if mature>=1420 & mature<1470
replace period=6 if mature>=1470 & mature<1520
replace period=7 if mature>=1520 & mature<1570
replace period=8 if mature>=1570 & mature<1620
replace period=9 if mature>=1620 & mature<1670
replace period=10 if mature>=1670 & mature<1720
replace period=11 if mature>=1720 & mature<1770
replace period=12 if mature>=1770 & mature<1820
drop if period == .
gen post = mature >= 1520
egen period_start = min(mature), by(period)
collapse (sum) people, by(cityid period* post protestant_at_t ind_law u1800 cityname polity1500  river hansa  longitude latitude  mean_plague1500_1521 mean_plague1400_1499)

gen trend = period - 6

gen post_x_law= post * ind_law
gen post_x_trend= post * trend
gen trend_x_law = ind_law * trend
gen laws_x_post_x_trend = ind_law * post * trend

tab period, gen(dec_dummy)

forvalues i=1(1)12{
gen law_x_decade_`i'=ind_law * dec_dummy`i' 
}

bysort cityid: gen counter=1 if _n==1
replace counter=sum(counter)

forvalues i=1(1)279{
gen city_trend_`i'=trend if counter==`i'
replace city_trend_`i'=0 if counter!=`i'
}

gen log_people=log(people+1)
tempfile data_formation
save `data_formation', replace

/*==============================================
STEP 2: CONSTRUCT DATA == DEATHS OF MIGRANTS
==============================================*/

// original version == migrant observed at death

clear all
set more off

* use 50 year intervals

use "$datadir/reformation_towns.dta", clear

drop if cityname_cantoni == ""
drop if markettown==""
drop if cityid==.
drop if u1800 == .

* start with cross-section
merge 1:m cityid using "$datadir/db_data_city_decade_final.dta", gen(_city)

collapse (sum) d_migrants_b, by(cityid decade protestant_at_t ind_law u1800 cityname polity1500  river hansa  longitude latitude mean_plague1500_1521 mean_plague1400_1499)

* assume maturity at age (decade) of death
gen mature = decade
gen period = 1 if mature>=1220 &mature<1270
replace period = 2 if mature>=1270 & mature<1320
replace period = 3 if mature>=1320 & mature<1370
replace period = 4 if mature>=1370 & mature<1420
replace period = 5 if mature>=1420 & mature<1470
replace period = 6 if mature>=1470 & mature<1520
replace period = 7 if mature>=1520 & mature<1570
replace period = 8 if mature>=1570 & mature<1620
replace period = 9 if mature>=1620 & mature<1670
replace period = 10 if mature>=1670 & mature<1720
replace period = 11 if mature>=1720 & mature<1770
replace period = 12 if mature>=1770 & mature<1820
drop if period == .
gen post = mature >= 1520
egen period_start = min(decade), by(period)
collapse (sum) d_migrants_b, by(cityid period* post protestant_at_t ind_law  u1800 cityname polity1500  river hansa  longitude latitude mean_plague1500_1521 mean_plague1400_1499)

gen trend = period - 6

gen post_x_law= post * ind_law
gen post_x_trend= post * trend
gen trend_x_law = ind_law * trend
gen laws_x_post_x_trend = ind_law * post * trend

tab period, gen(dec_dummy)

forvalues i=1(1)12{
gen law_x_decade_`i' = ind_law * dec_dummy`i' 
}

drop law_x_decade_6 

bysort cityid: gen counter = 1 if _n==1
replace counter=sum(counter)

forvalues i=1(1)279{
gen city_trend_`i' = trend if counter==`i'
replace city_trend_`i' = 0 if counter!=`i'
}

gen log_migrants=log(d_migrants_b + 1)
tempfile data_migrants
save `data_migrants', replace

// updated version == "migrant 20 years before death"

clear all
set more off

* use 50 year intervals

use "$datadir/reformation_towns.dta", clear

drop if cityname_cantoni == ""
drop if markettown==""
drop if cityid==.
drop if u1800 == .

* start with cross-section
merge 1:m cityid using "$datadir/db_data_city_decade_final.dta", gen(_city)

ren d_migrants_b d_migrants_b2

* reassign migrants to 20 years before death
replace decade = decade - 20
collapse (sum) d_migrants_b2, by(cityid decade protestant_at_t ind_law u1800 cityname polity1500  river hansa  longitude latitude markettownid)

* assume maturity two decades before (decade) of death == as per assignment above
gen mature = decade
gen period=1 if mature>=1220 &mature<1270
replace period=2 if mature>=1270 & mature<1320
replace period=3 if mature>=1320 & mature<1370
replace period=4 if mature>=1370 & mature<1420
replace period=5 if mature>=1420 & mature<1470
replace period=6 if mature>=1470 & mature<1520
replace period=7 if mature>=1520 & mature<1570
replace period=8 if mature>=1570 & mature<1620
replace period=9 if mature>=1620 & mature<1670
replace period=10 if mature>=1670 & mature<1720
replace period=11 if mature>=1720 & mature<1770
replace period=12 if mature>=1770 & mature<1820
drop if period == .
gen post = mature >= 1520
egen period_start = min(decade), by(period)
collapse (sum) d_migrants_b2, by(cityid period* post protestant_at_t ind_law  u1800 cityname polity1500  river hansa  longitude latitude markettownid)

gen trend=period

gen post_x_law= post * ind_law
gen post_x_trend= post * trend
gen trend_x_law = ind_law * trend
gen laws_x_post_x_trend = ind_law * post * trend

tab period, gen(dec_dummy)

forvalues i=1(1)12{
gen law_x_decade_`i'=ind_law * dec_dummy`i' 
}

drop law_x_decade_6 

bysort cityid: gen counter=1 if _n==1
replace counter=sum(counter)

forvalues i=1(1)279{
gen city_trend_`i'=trend if counter==`i'
replace city_trend_`i'=0 if counter!=`i'
}

keep period_start cityid d_migrants_b2 markettownid
gen log_migrants2 = log(d_migrants_b2 + 1)
tempfile data_migrants2
save `data_migrants2', replace

use `data_migrants', clear
merge m:m cityid period_start using `data_migrants2'

/*==============================================
STEP 3 == COMBINE DATA
==============================================*/

* use `data_migrants', clear
drop _merge
merge 1:1 cityid period using `data_formation', gen(_merge3)
drop if period == .

gen total = d_migrants_b + people
gen ln_total = ln(total)
gen ln_total_p1 = ln(total + 1)

/*==============================================
STEP 4 == CLEAN UP
==============================================*/

lab var post_x_law "Post $\times$ Law" 
lab var post_x_trend "Post $\times$ Trend"
lab var trend "Trend" 
lab var laws_x_post_x_trend "Post $\times$ Trend $\times$ Law"
lab var trend_x_law "Trend $\times$ Law"  

gen any_migrants = d_migrants_b > 0
gen any_people =  people > 0
egen any_total = rowmax(any_people any_migrants)

encode polity1500, gen(polity1500_num)
egen territory_x_year = group(polity1500 period)

lab var log_people "Native"
lab var log_migrants "Migrant"
lab var any_migrants "Migrant"
lab var any_people "Native"

egen people_mean_period = mean(people), by(period_start)
gen people_above_mean = people > people_mean_period & people != .
egen migrant_mean_period = mean(d_migrants_b), by(period_start)
gen migrant_above_mean = d_migrants_b > migrant_mean_period & d_migrants_b != .

* plague variables
gen p_exc1500_1521 = 22*(mean_plague1500_1521 - mean_plague1400_1499) 
gen p_any1500_1521 = mean_plague1500_1521 > 0
local list "p_exc1500_1521 p_any1500_1521"
foreach x in `list' {
	gen post_x_`x' = post * `x' 
	gen trend_x_`x'  = `x' * trend
	gen `x'_x_post_x_trend = `x'  * post * trend
}

lab var post_x_p_exc1500_1521 "Post $\times$ Plague Shock" 
lab var p_exc1500_1521_x_post_x_trend "Post $\times$ Trend $\times$ Plague Shock"
lab var trend_x_p_exc1500_1521 "Trend $\times$ Plague Shock"  

lab var post_x_p_any1500_1521 "Post $\times$ Plague 1500-1521" 
lab var p_any1500_1521_x_post_x_trend "Post $\times$ Trend $\times$ Plague 1500-1521"
lab var trend_x_p_any1500_1521 "Trend $\times$ Plague 1500-1521"  

drop _m
tempfile temp
save `temp', replace

* data == identify free cities for principalities
use "$datadir/reformation_towns.dta", clear
keep if markettownid != . & cityid != . & cityname_cantoni != "" & ln_u1800 != .
keep markettownid ind_freecity
tempfile data_freecity
save `data_freecity', replace

use `temp', clear
gen century = 100 * floor(period_start/100)
merge m:1 markettownid using `data_freecity'
drop _merge

* data work for referee #3 == principalities
drop if markettownid == .
merge m:1 markettownid using "$datadir/reformation_towns_territories.dta"

qui tab terr_id_1500, gen(terr1500_)
gen terr_id_1500alt = terr_id_1500
replace terr_id_1500alt = "FREE CITY" if ind_freecity == 1 
qui tab terr_id_1500alt, gen(terr1500alt_)
egen principality_x_year = group(terr_id_1500alt period)

gen post_x_prot = post * protestant_at_t
gen post_x_prot_x_trend = post * protestant_at_t * trend

set matsize 10000
drop if cityid == .

/*==============================================
RESPONSE TO R3
	NATURE OF COUNTERFACTUAL
	EXCLUDE PRINCIPALITIES WITH ANY RELIGIOUS
	HETEROGENEITY
==============================================*/

* data == identify principalities with variation in religion
egen total_prot = sum(protestant_at_t), by(terr_id_1500alt period_start)
gen d = 1
egen total_cities = sum(d), by(terr_id_1500alt period_start)
gen heterogeneous = total_prot < total_cities & total_prot != 0
drop d

* data == identify principalities with variation in law
egen total_law = sum(ind_law), by(terr_id_1500alt period_start)
gen law_heterogeneous = total_law < total_cities & total_law != 0

tab terr_id_1500alt if law_heterogeneous == 1 & period_start == 1520 & heterogeneous == 0
tab ind_law if law_heterogeneous == 1 & period_start == 1520 & heterogeneous == 0
tab terr_id_1500alt if law_heterogeneous == 1 & period_start == 1520 & heterogeneous == 0

/*==============================================
SAVE PREPARED DATASET FOR TABLE 2
==============================================*/

* Save the prepared dataset just before regressions
save "$datadir/table2_prepared_data.dta", replace
display "Dataset for Table 2 saved as: $datadir/table2_prepared_data.dta"

