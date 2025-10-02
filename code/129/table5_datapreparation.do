/*=========================
TABLE 5
=========================*/

clear all

* Set the data directory
global datadir "/Users/kimik/Desktop/RA (retransformation bias)/papers to be replicated/129/Datafortheweb/data"

* data
use "$datadir/reformation_towns.dta", clear
drop if markettownid==.

* data == principalities
merge 1:1 markettownid using "$datadir/reformation_towns_territories.dta"

* == upper tail human capital measures == *

* total based on age of maturity == 50-YEAR BINS ON 20'S AND 70'S
gen db_total1770 = db_form_mat50_1770 + db_migrants50_1770
gen db_total1470 = db_form_mat50_1470 + db_migrants50_1470
gen db_total1420 = db_form_mat50_1420 + db_migrants50_1420
gen ln_db_total_p1 = ln(db_total1770 + 1)
lab var ln_db_total_p1 "Ln Total"
gen ln_db_total1770 = ln(db_total1770 + 1)
gen ln_db_total1470 = ln(db_total1470 + 1)
gen ln_db_total1420 = ln(db_total1420 + 1)

* total based on age of maturity == 50-YEAR BINS ON HALF CENTURIES
gen db_total1750 = db_form_mat50_1750 + db_migrants50_1750
gen db_total1450 = db_form_mat50_1450 + db_migrants50_1450
gen db_total1400 = db_form_mat50_1400 + db_migrants50_1400
gen ln_db_total1750 = ln(db_total1750 + 1)
gen ln_db_total1450 = ln(db_total1450 + 1)
gen ln_db_total1400 = ln(db_total1400 + 1)

* share based on age of maturity == 50-YEAR BINS ON HALF CENTURIES
gen shr_tot1750 = (db_form_mat50_1750 + db_migrants50_1750) /u1800

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
egen mean_plague1349_1499 = rowmean(ind_plague1349-ind_plague1499)

* latitude-longitude interaction
gen geo_interact=latitude * longitude

* ln 1500 pop
gen ln_variable = ln(u1500)
replace ln_variable = 0 if ln_variable == .
	
* labels
lab var ind_law "Reformation Law"
lab var pop1500_0 "Population 1500: Unobserved"
lab var pop1500_1_5 "Population 1500: 1-5"
lab var pop1500_6_10 "Population 1500: 6-10"
lab var pop1500_11_20 "Population 1500: 11-20"
lab var pop1500_21 "Population 1500: 21+"
lab var protestant_at_t "Protestant"
lab var distance_wittenberg "Distance to Wittenberg"

* Rename the merge variable from territories merge
capture ren _merge merge_territories

qui tab terr_id_1500, gen(terr1500_)
gen terr_id_1500alt = terr_id_1500
replace terr_id_1500alt = "FREE CITY" if ind_freecity == 1 
egen principality_x_year = group(terr_id_1500alt year)

* data == identify principalities with religious heterogeneity
egen total_prot = sum(protestant_at_t), by(terr_id_1500alt)
capture drop d  // Drop d if it exists
gen d = 1  // Create indicator variable
egen total_cities = sum(d), by(terr_id_1500alt)
gen heterogeneous = total_prot < total_cities & total_prot != 0
drop d  // Clean up temporary variable

/*==============================================
SAVE PREPARED DATASET FOR TABLE 5
==============================================*/

* Save the prepared dataset just before regressions
save "$datadir/table5_prepared_data.dta", replace
display "Dataset for Table 5 saved as: $datadir/table5_prepared_data.dta"

eststo clear
