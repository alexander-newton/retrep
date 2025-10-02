/*=========================
TABLE 6
=========================*/

clear all

* Set the data directory
global datadir "/Users/kimik/Desktop/RA (retransformation bias)/papers to be replicated/129/Datafortheweb/data"

* data
use "$datadir/reformation_towns.dta", clear
	
/*=======================================================
VARIABLES
=======================================================*/

* == upper tail human capital measures == *

* total based on age of maturity
gen db_total1770 = db_form_mat50_1770 + db_migrants50_1770
gen db_total1750 = db_form_mat50_1750 + db_migrants50_1750
gen db_total1470 = db_form_mat50_1470 + db_migrants50_1470
gen db_total1420 = db_form_mat50_1420 + db_migrants50_1420
gen ln_db_total_p1 = ln(db_total1770 + 1)
lab var ln_db_total_p1 "Ln Total"
gen ln_db_total1770 = ln(db_total1770 + 1)
gen ln_db_total1750 = ln(db_total1750 + 1)
gen ln_db_total1470 = ln(db_total1470 + 1)
gen ln_db_total1420 = ln(db_total1420 + 1)
gen shr_tot1750 = (db_form_mat50_1750 + db_migrants50_1750) /u1800

* total based on birth
gen db_tot_b1770 = db_form_born50_1770 + db_migrants50_1770
gen db_tot_b1470 = db_form_born50_1470 + db_migrants50_1470
gen db_tot_b1420 = db_form_born50_1420 + db_migrants50_1420
gen ln_db_tot_b_p1 = ln(db_tot_b1770 + 1)
lab var ln_db_tot_b_p1 "Ln Total"
gen ln_db_tot_b1770 = ln(db_tot_b1770 + 1)
gen ln_db_tot_b1470 = ln(db_tot_b1470 + 1)
gen ln_db_tot_b1420 = ln(db_tot_b1420 + 1)

* formation == mature
gen ln_db_form_mat50_1770 = ln(db_form_mat50_1770 + 1)
gen ln_db_form_mat50_1470 = ln(db_form_mat50_1470 + 1)
gen ln_db_form_mat50_1420 = ln(db_form_mat50_1420 + 1)

* formation == born
gen ln_db_form_born50_1770 = ln(db_form_born50_1770 + 1)
gen ln_db_form_born50_1470 = ln(db_form_born50_1470 + 1)
gen ln_db_form_born50_1420 = ln(db_form_born50_1420 + 1)

* migrants
gen ln_db_migrants50_1770 = ln(db_migrants50_1770 + 1)
gen ln_db_migrants50_1470 = ln(db_migrants50_1470 + 1)
gen ln_db_migrants50_1420 = ln(db_migrants50_1420 + 1)

* students from city X going to university 1498-1517
gen studs = students10_1498 + students10_1508

* normalized "plagues over 23 year windows"
gen plagues_years1500_1522 = 23* mean_plague1500_1522
gen plagues_any1500_1522 = mean_plague1500_1522 > 0 & mean_plague1500_1522 != .
gen plagues_norm1400_1499 = plague_years1400_1499 / 23
gen plagueany1400 = mean_plague1400_1499 != 0
egen plagues_years1500_1550 = rowtotal(ind_plague1500-ind_plague1550)

* plague periods
forvalues x = 1400(25)1500 {
	local y = `x' + 24
	egen plague`x'_`y' = rowtotal(ind_plague`x'-ind_plague`y')
}

* treatment variable 
gen test = ind_law
gen X1 = mean_plague1400_1499
gen X2 = mean_plague1400_1499^2
gen X3 = mean_plague1400_1499^3
lab var X1 "Level - Mean Plague 1400-1499"
lab var X2 "Quadratic - Mean Plague 1400-1499"
lab var X3 "Cubic - Mean Plague 1400-1499"

* data work for referee #3 == principalities
drop if markettownid == .
merge 1:1 markettownid using "$datadir/reformation_towns_territories.dta"
drop if _m == 1

qui tab terr_id_1500, gen(terr1500_)
gen terr_id_1500alt = terr_id_1500
replace terr_id_1500alt = "FREE CITY" if ind_freecity == 1 
qui tab terr_id_1500alt, gen(terr1500alt_)

* plagues in 1300s == new measures of plague including 1300s
egen mean_plague1349_1499 = rowmean(ind_plague1349-ind_plague1499)
egen plague1350_1374 = rowtotal(ind_plague1350-ind_plague1374)
egen plague1375_1399 = rowtotal(ind_plague1375-ind_plague1399)

/*==============================================
SAVE PREPARED DATASET FOR TABLE 6
==============================================*/

* Save the prepared dataset just before regressions
save "$datadir/table6_prepared_data.dta", replace
display "Dataset for Table 6 saved as: $datadir/table6_prepared_data.dta"

/*=======================================================
Table 6, first panel
IV REGRESSIONS == PANEL A
	FIRST STAGE REGRESSIONS == BASE CONTROLS FOR 1500 POP
	ALSO CONTROLS FOR DB
	CONTROLS FOR PLAGUES IN 1300S
=======================================================*/

// mean_plague1349_1499 plague1350_1374 plague1375_1399
eststo clear

