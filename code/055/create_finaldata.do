
clear
capture log close
set more off
set scheme s1color
version 14.0

******************************************
* CHANGE PATH BASED ON THE COMPUTER NAME *
******************************************

if "`c(hostname)'" == "IT029423" { // Stephan Laptop
	cd "C:/Users/sh13682/Dropbox/MakingMetropolis/QJEaccepted/MMM/stata/"
	adopath + "/Users/sh13682/Dropbox/MakingMetropolis/QJEaccepted/MMM/stata/ado"
}
if "`c(hostname)'" == "IT062630" { // Stephan Desktop
	cd "C:/Users/sh13682/Dropbox/MakingMetropolis/QJEaccepted/MMM/stata/"
	adopath + "/Users/sh13682/Dropbox/MakingMetropolis/QJEaccepted/MMM/stata/ado"
}
if "`c(username)'" == "reddings" { // SJR mac laptop
	cd "/Users/reddings/Dropbox/MakingMetropolis/QJEaccepted/MMM/stata/"
	adopath + "/Users/reddings/Dropbox/MakingMetropolis/QJEaccepted/MMM/stata/ado"
}
if "`c(username)'" == "kimik" { // Your computer
	cd "/Users/kimik/Desktop/RA (retransformation bias)/papers to be replicated/55/stata/"
	* adopath + "/Users/kimik/Desktop/RA (retransformation bias)/papers to be replicated/55/stata/ado"
}

*log using "code/4-gravity-estim/gravity.log" , replace

****************************************************************
* PREPARE ARCGIS BOROUGH RANDOM POINT FULL NETWORK TRAVEL TIME *
****************************************************************

u data/arc-gis-data/Borough_GLA_TTRANS_RP_1831_1921_final.dta , clear
keep G_UNIT_o G_UNIT_d ttrans_RP1921
ren ttrans_RP1921 ttrans1921
so G_UNIT_o G_UNIT_d 
sa data/ttrans_arcgis_bor1921.dta , replace

*********************************************************
* CLEAN ARCGIS BOROUGH RANDOM POINT WALKING TRAVEL TIME *
*********************************************************

u data/arc-gis-data/Borough_GLA_TT_RP_final.dta , clear
ren tt_RP tt1921
keep G_UNIT_o G_UNIT_d tt1921
so G_UNIT_o G_UNIT_d 
sa data/tt_arcgis_bor1921.dta , replace

*****************
* CLEAN DIST XY *
*****************

clear
insheet G_UNIT_o G_UNIT_d distxy1921 using ../matlab/data/output/distxy1921.csv , nonames
ren g_unit_o G_UNIT_o
ren g_unit_d G_UNIT_d
lab var distxy1921 	"X-Y Distance Between Borough Centroids"
isid G_UNIT_o G_UNIT_d 
so G_UNIT_o G_UNIT_d 
sa data/distxy1921.dta, replace

***************************************
* CENTRAL ORIGIN AND DESTINATION DATA *
***************************************

clear
insheet G_UNIT_o G_UNIT_d cen_o using ../matlab/data/output/cen_o.csv , nonames
ren g_unit_o G_UNIT_o
ren g_unit_d G_UNIT_d
lab var cen_o "cen_o==1 for origins in Central London (Holborn, City of London & Westminster)"
isid G_UNIT_o G_UNIT_d 
so G_UNIT_o G_UNIT_d 
sa data/cen_o.dta, replace

clear
insheet G_UNIT_o G_UNIT_d cen_d using ../matlab/data/output/cen_d.csv , nonames
ren g_unit_o G_UNIT_o
ren g_unit_d G_UNIT_d
lab var cen_d "cen_d==1 for destinations in Central London (Holborn, City of London & Westminster)"
isid G_UNIT_o G_UNIT_d 
so G_UNIT_o G_UNIT_d 
sa data/cen_d.dta, replace

*****************************************
* CLEAN LCC ORIGIN AND DESTINATION DATA *
*****************************************

clear
insheet G_UNIT_o G_UNIT_d othlcc_o using ../matlab/data/output/othlcc_o.csv , nonames
ren g_unit_o G_UNIT_o
ren g_unit_d G_UNIT_d
lab var othlcc_o "othlcc_o==1 for origins in parts of County of London outside Central London"
isid G_UNIT_o G_UNIT_d 
so G_UNIT_o G_UNIT_d 
sa data/othlcc_o.dta, replace

clear
insheet G_UNIT_o G_UNIT_d othlcc_d using ../matlab/data/output/othlcc_d.csv , nonames
ren g_unit_o G_UNIT_o
ren g_unit_d G_UNIT_d
lab var othlcc_d "othlcc_d==1 for destinations in parts of County of London outside Central London"
isid G_UNIT_o G_UNIT_d 
so G_UNIT_o G_UNIT_d 
sa data/othlcc_d.dta, replace

*****************************************
* CLEAN GLA ORIGIN AND DESTINATION DATA *
*****************************************

clear
insheet G_UNIT_o G_UNIT_d othgla_o using ../matlab/data/output/othgla_o.csv , nonames
ren g_unit_o G_UNIT_o
ren g_unit_d G_UNIT_d
lab var othgla_o "othgla_o==1 for origins in parts of GLA outside of County of London"
isid G_UNIT_o G_UNIT_d 
so G_UNIT_o G_UNIT_d 
sa data/othgla_o.dta, replace

clear
insheet G_UNIT_o G_UNIT_d othgla_d using ../matlab/data/output/othgla_d.csv , nonames
ren g_unit_o G_UNIT_o
ren g_unit_d G_UNIT_d
lab var othgla_d "othgla_d==1 for destinations in parts of GLA outside of County of London"
isid G_UNIT_o G_UNIT_d 
so G_UNIT_o G_UNIT_d 
sa data/othgla_d.dta, replace

********************
* PREPARE X-Y DATA *
********************

clear
u data/bilateral-commuting-data/GLA1921_Borough_Centroids_MATLAB.dta
ren G_UNIT G_UNIT_o
ren POINT_X X_o
ren POINT_Y Y_o
keep G_UNIT_o X_o Y_o 
lab var X_o "X coordinate origin"
lab var Y_o "Y coordinate origin"
so G_UNIT_o
sa data/XY_o.dta, replace

ren G_UNIT_o G_UNIT_d
ren X_o X_d
ren Y_o Y_d
lab var X_d "X coordinate destination"
lab var Y_d "Y coordinate destination"
so G_UNIT_d
sa data/XY_d.dta, replace

**************************************
* PREPARE DISTANCE TO GUILDHALL DATA *
**************************************

clear
u data/GLA_Centroid_DistGuildhall.dta
ren G_UNIT G_UNIT_o
ren dist_guildhall_km distguild_o
keep G_UNIT_o distguild_o
lab var distguild_o "Distance of origin from Guildhall"
so G_UNIT_o
sa data/distguild_o.dta, replace

ren G_UNIT_o G_UNIT_d
ren distguild_o distguild_d
lab var distguild_d "Distance of destination from Guildhall"
so G_UNIT_d
sa data/distguild_d.dta, replace

*************************
* 1921 COMMUTING MATRIX *
*************************

u data/commutematrix1921_final_GUNIT.dta, clear

*****************************
* MERGE ARCGIS TRAVEL TIMES *
*****************************

so G_UNIT_o G_UNIT_d 
merge 1:1 G_UNIT_o G_UNIT_d using data/ttrans_arcgis_bor1921.dta
tab _m
drop _m

so G_UNIT_o G_UNIT_d 
merge 1:1 G_UNIT_o G_UNIT_d using data/tt_arcgis_bor1921.dta
tab _m
drop _m

*********************
* MERGE XY DISTANCE *
*********************

so G_UNIT_o G_UNIT_d 
merge 1:1 G_UNIT_o G_UNIT_d using data/distxy1921.dta
tab _m
drop _m

*************************************
* MERGE CEN, LCC AND GLA INDICATORS *
*************************************

so G_UNIT_o G_UNIT_d 
merge 1:1 G_UNIT_o G_UNIT_d using data/cen_o.dta
tab _m
drop _m

so G_UNIT_o G_UNIT_d 
merge 1:1 G_UNIT_o G_UNIT_d using data/cen_d.dta
tab _m
drop _m

so G_UNIT_o G_UNIT_d 
merge 1:1 G_UNIT_o G_UNIT_d using data/othlcc_o.dta
tab _m
drop _m

so G_UNIT_o G_UNIT_d 
merge 1:1 G_UNIT_o G_UNIT_d using data/othlcc_d.dta
tab _m
drop _m

so G_UNIT_o G_UNIT_d 
merge 1:1 G_UNIT_o G_UNIT_d using data/othgla_o.dta
tab _m
drop _m

so G_UNIT_o G_UNIT_d 
merge 1:1 G_UNIT_o G_UNIT_d using data/othgla_d.dta
tab _m
drop _m

gen test_o=cen_o+othlcc_o+othgla_o
tab test_o
drop test_o

gen test_d=cen_d+othlcc_d+othgla_d
tab test_d
drop test_d

gen cen_o_cen_d=cen_o*cen_d
gen othlcc_o_cen_d=othlcc_o*cen_d
gen othgla_o_cen_d=othgla_o*cen_d

gen cen_o_othlcc_d=cen_o*othlcc_d
gen othlcc_o_othlcc_d=othlcc_o*othlcc_d
gen othgla_o_othlcc_d=othgla_o*othlcc_d

gen cen_o_othgla_d=cen_o*othgla_d
gen othlcc_o_othgla_d=othlcc_o*othgla_d
gen othgla_o_othgla_d=othgla_o*othgla_d

lab var cen_o_cen_d 		"cen_o*cen_d"
lab var othlcc_o_cen_d		"othlcc_o*cen_d"
lab var othgla_o_cen_d		"othgla_o*cen_d"

lab var cen_o_othlcc_d		"cen_o*othlcc_d"
lab var othlcc_o_othlcc_d	"othlcc_o*othlcc_d"
lab var othgla_o_othlcc_d	"othgla_o*othlcc_d"

lab var cen_o_othgla_d		"cen_o*othgla_d"
lab var othlcc_o_othgla_d	"othlcc_o*othgla_d"
lab var othgla_o_othgla_d	"othgla_o*othgla_d"

so G_UNIT_o G_UNIT_d 

************************************************
* MERGE ORIGIN AND DESTINATION CHARACTERISTICS *
************************************************

so G_UNIT_o G_UNIT_d
merge m:1 G_UNIT_o using data/XY_o.dta
tab _m
drop _m

so G_UNIT_d G_UNIT_o
merge m:1 G_UNIT_d using data/XY_d.dta
tab _m
drop _m

so G_UNIT_o G_UNIT_d
merge m:1 G_UNIT_o using data/distguild_o.dta
tab _m
drop _m

so G_UNIT_d G_UNIT_o
merge m:1 G_UNIT_d using data/distguild_d.dta
tab _m
drop _m

so G_UNIT_o G_UNIT_d

***********************
* SHARES OF COMMUTERS *
***********************

preserve

drop if G_UNIT_o==G_UNIT_d

gen incommute=flow

collapse (sum) incommute (mean) distguild_d , by(G_UNIT_d destination)

lab var incommute "Incommuters into a destination from other origins (excluding self)"
lab var distguild_d "Distance of destination from Guildhall"

gsort -incommute
gen order=_n

lab var order "Rank order of observations sorted by incommuters"

list if distguild_d<5

egen tot_incommute=sum(incommute) 
gen incommshare=incommute/tot_incommute

lab var tot_incommute "Total incommuters for Greater London"
lab var incommshare "Incommuters from other origins as a share of total incommuters for Greater London"

keep if distguild_d<5

collapse (sum) incommshare
list

*twoway (scatter incommshare distguild_d) ///
*, xline(5)

restore

***********************
* TRANSFORM VARIABLES *
***********************

egen tot_flow=sum(flow)
gen uprob=flow/tot_flow

gen lflow=ln(flow)
gen luprob=ln(uprob)
gen ldistxy1921=ln(distxy1921)
gen ltt1921=ln(tt1921)
gen lttrans1921=ln(ttrans1921)

lab var tot_flow 		"sum(flow)"
lab var uprob 			"Probability of commuting from origin to destination in Greater London"
lab var lflow			"ln(flow)"
lab var luprob			"ln(uprob)"
lab var ldistxy1921		"ln(distxy1921)"
lab var ltt1921			"ln(tt1921)"
lab var lttrans1921		"ln(ttrans1921)"

gen zone=1 if cen_o==1&cen_d==1
replace zone=2 if othlcc_o==1&cen_d==1
replace zone=3 if othgla_o==1&cen_d==1

replace zone=4 if cen_o==1&othlcc_d==1
replace zone=5 if othlcc_o==1&othlcc_d==1
replace zone=6 if othgla_o==1&othlcc_d==1

replace zone=7 if cen_o==1&othgla_d==1
replace zone=8 if othlcc_o==1&othgla_d==1
replace zone=9 if othgla_o==1&othgla_d==1

lab def zone 1 "cen_o==1&cen_d==1" 2 "othlcc_o==1&cen_d==1" 3 "othgla_o==1&cen_d==1" 4 "cen_o==1&othlcc_d==1" 5 "othlcc_o==1&othlcc_d==1" 6 "othgla_o==1&othlcc_d==1" 7 "cen_o==1&othgla_d==1" 8 "othlcc_o==1&othgla_d==1" 9 "othgla_o==1&othgla_d"
lab val zone zone 
lab var zone "Categorical variable (1-9) for bilateral pairs of zones in Greater London"

gen ldistxy1921sq=ldistxy1921*ldistxy1921
gen ltt1921sq=ltt1921*ltt1921

xtile xltt1921 = ltt1921 , nq(5)

lab var ldistxy1921sq 		"ldistxy1921*ldistxy1921"
lab var ltt1921sq  			"ltt1921*ltt1921"
lab var xltt1921			"ltt1921 , nq(5)"

*****************
* SELECT SAMPLE *
*****************

gen sample=1
replace sample=0 if G_UNIT_o==G_UNIT_d
replace sample=0 if flow==0


lab var sample "sample==1 for positive commuting flows to other destinations (excluding self)"

save "data/gravity_analysis_final.dta", replace
display "Final dataset saved with " _N " observations


/*

****************************
* TABLE 1 GRAVITY IN PAPER *
****************************

eststo clear

* TOP PANEL

* Column (1) : OLS
eststo: reghdfe luprob lttrans1921 if sample==1 , absorb(i.G_UNIT_o i.G_UNIT_d) vce(robust)

* Column (2) : Second-stage IV
eststo: ivreghdfe luprob (lttrans1921 = ltt1921) if sample==1 , absorb(i.G_UNIT_o i.G_UNIT_d) robust

local phieps=_b[lttrans1921]
display "`phieps'"

* BOTTOM PANEL

* Column (2) : First-stage IV

eststo: reghdfe lttrans1921 ltt1921 if sample==1 , absorb(i.G_UNIT_o i.G_UNIT_d) vce(robust)
test ltt1921

esttab using results/table_I_gravity.csv, b(%9.3f) se(%9.3f) star(* 0.10 ** 0.05 *** 0.01) r2 replace

*******************************
* TABLE I4 IN ONLINE APPENDIX *
*******************************

su ltt1921 if sample==1, d
gen med_ltt1921=r(p50)

eststo clear

* TOP PANEL

* Column (1) : Second-stage IV, all straight-line distance
eststo: ivreghdfe luprob (lttrans1921 = ltt1921) if sample==1 , absorb(i.G_UNIT_o i.G_UNIT_d) robust

* Column (2) : Second-stage IV, below median straight-line distance
eststo: ivreghdfe luprob (lttrans1921 = ltt1921) if (ltt1921<med_ltt1921)&(sample==1) , absorb(i.G_UNIT_o i.G_UNIT_d) robust

* Column (3) : Second-stage IV, above median straight-line distance
eststo: ivreghdfe luprob (lttrans1921 = ltt1921) if (ltt1921>=med_ltt1921)&(sample==1) , absorb(i.G_UNIT_o i.G_UNIT_d) robust

* Column (4) : Second-stage IV, quadratic straight-line distance
eststo: ivreghdfe luprob (lttrans1921 = ltt1921 ltt1921sq) if sample==1 , absorb(i.G_UNIT_o i.G_UNIT_d) robust

* Column (5) : Second-stage IV, quadratic straight-line distance, zone pair fixed effects
eststo: ivreghdfe luprob (lttrans1921 = ltt1921 ltt1921sq) if sample==1 , absorb(i.G_UNIT_o i.G_UNIT_d i.zone) robust

* Column (6) : Second-stage IV, quadratic straight-line distance, zone pair and straight-line distance quintile fixed effects
eststo: ivreghdfe luprob (lttrans1921 = ltt1921 ltt1921sq) if sample==1 , absorb(i.G_UNIT_o i.G_UNIT_d i.zone i.xltt1921) robust

* BOTTOM panel

* Column (1) : First-stage IV, all straight-line distance
eststo: reghdfe lttrans1921 ltt1921 if sample==1 , absorb(i.G_UNIT_o i.G_UNIT_d) vce(robust)
test ltt1921 

* Column (2) : First-stage IV, below median straight-line distance
eststo: reghdfe lttrans1921 ltt1921 if (ltt1921<med_ltt1921)&(sample==1) , absorb(i.G_UNIT_o i.G_UNIT_d) vce(robust)
test ltt1921

* Column (3) : First-stage IV, above median straight-line distance
eststo: reghdfe lttrans1921 ltt1921 if (ltt1921>=med_ltt1921)&(sample==1) , absorb(i.G_UNIT_o i.G_UNIT_d) vce(robust)
test ltt1921

* Column (4) : First-stage IV, quadratic straight-line distance
eststo: reghdfe lttrans1921 ltt1921 ltt1921sq if sample==1 , absorb(i.G_UNIT_o i.G_UNIT_d) vce(robust)
test ltt1921 ltt1921sq

* Column (5) : First-stage IV, quadratic straight-line distance, zone pair fixed effects
eststo: reghdfe lttrans1921 ltt1921 ltt1921sq if sample==1 , absorb(i.G_UNIT_o i.G_UNIT_d i.zone) vce(robust)
test ltt1921 ltt1921sq

* Column (6) : First-stage IV, quadratic straight-line distance, zone pair and straight-line distance quintile fixed effects
eststo: reghdfe lttrans1921 ltt1921 ltt1921sq if sample==1 , absorb(i.G_UNIT_o i.G_UNIT_d i.zone i.xltt1921) vce(robust)
test ltt1921 ltt1921sq

esttab using results/table_I4_gravity_online_appendix.csv, b(%9.3f) se(%9.3f) star(* 0.10 ** 0.05 *** 0.01) r2 replace

*************************************
* POISSON PSEUDO MAXIMUM LIKELIHOOD *
*************************************

glm uprob lttrans1921 i.G_UNIT_o i.G_UNIT_d , family(poisson) link(log) vce(robust) difficult

ivpoisson gmm uprob (lttrans1921 = ltt1921 ltt1921sq) i.G_UNIT_o i.G_UNIT_d  , vce(robust) 

********************
* Estimated phieps *
******************** 

display ">>>> Estimated Phi Epsilon <<<<"
display "`phieps'"

gen kappa=ttrans1921^(`phieps')
gen lkappa=ln(kappa)

****************************************
* Conditional correlation second-stage *
****************************************

gen mlkappa=-lkappa

reghdfe luprob if sample==1 , absorb(FEo=i.G_UNIT_o FEd=i.G_UNIT_d) vce(robust) resid
predict rluprob if sample==1 , res
drop FEo FEd

reghdfe mlkappa if sample==1 , absorb(FEo=i.G_UNIT_o FEd=i.G_UNIT_d) vce(robust) resid
predict rmlkappa if sample==1 , res
drop FEo FEd

reg rluprob rmlkappa if sample==1 , robust
predict frmluprob if sample==1 , xb

twoway (scatter rluprob rmlkappa if sample==1 , msymbol(o) mcolor(blue)) ///
(line frmluprob rmlkappa if sample==1 , msymbol(i) c(l) lcolor(red)) ///
, ///
legend(off) ///
xtitle(Residual Log Commuting Costs, size(medium)) ///
ytitle(Residual Log Commuting Probabilities, size(medium)) 

graph export "code/4-gravity-estim/figures/Figure_I19_online_app_gravity.pdf", as(pdf) replace

drop rluprob rmlkappa frmluprob

log close
