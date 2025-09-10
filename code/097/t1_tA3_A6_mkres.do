*uses version of reghdfe 3.2.9 (21.2.16)
clear
version 14.0
mata: mata clear
set matsize 11000
macro drop _all
matrix drop _all
program drop _all
*global dir "/Users/ugte195/Dropbox (QMUL-SEF)/jdrive/papers"
*global dir "/Users/andreatesei/Dropbox"
global dir "O:\tew222\box\Dropbox"
*global dir "/home/ugte195"
global dir_inter "$dir/MOBILE PHONES/ECMA/final_submission/analysis"


cap program drop initial
program define initial
u "$dir_inter/original_data/main_data_final.dta", replace

*creates  vars
local group gdelt gdelt_opd acled acled_opd scad scad_opd gdelt_prec acled_prec scad_prec
foreach c of local group{
gen `c'_pc=(`c'/pop_ip)*100000
replace `c'_pc=. if gdelt_pc==.
gen ln_`c'_pc=ln(1+`c'_pc)
}

g ln_pop_ip=log(pop_ip)

*dummies for relevant variables
cap drop dcoast
gen dcoast=(coastdist==0)
local group mines diam
foreach c of local group{
gen d`c'=(`c'>0)
replace d`c'=. if `c'==.
}


*key interactions
g covgdp_wb=pct_cov*gdp_g_wb
g covgdp_wb_w=pct_cov_w*gdp_g_wb
 
 
*instrument
g flash2_tv=flash2*tv
g flash2_tvgdp=flash2_tv*gdp_g_wb
************


*missing values for covariates
global vars  coastdist frst mnt mines diam dmines ddiam imr dcoast bdist2 capdist electr prim_road  
 local group $vars
foreach c of local group{
g d_`c'=`c'==.
replace `c'=-999999 if `c'==.
}


*dummy mines and diamonds enter as factor variables and have missing values. Must be positive  
replace dmines=abs(dmines)
replace ddiam=abs(ddiam)

g border=(bdist2<.25)


*night lights per capita
g nl3_mp_pc=nl3_mp/pop_ip
g ln_nl3_mp_pc=ln(nl3_mp_pc)

*defines working sample
drop if year<1998
drop if ln_gdelt_pc==. | pct_cov==. | flash2_tv==. | covgdp_wb==.


*defines macros 
global sizecontrols2="c.ln_pop_ip"
global nl="c.ln_nl3_mp_pc"
for any prec_gpcp temp_prio: replace X=ln(X)   
global climate="c.prec_gpcp c.temp_prio"   

global baseline1 border c.bdist2  c.pct_oil  c.cities c.frst d_frst c.mnt d_mnt dmines d_dmines ddiam d_ddiam c.imr d_imr
global baseline2 capital c.capdist c.coastdist c.electr d_electr c.prim_road d_prim_road 
global otherbaseline dcoast c.lat c.lon c.totarea  
global baseline $baseline1 $baseline2 $otherbaseline 

end


cap progra drop regnowA
program define regnowA
local group ln_gdelt_pc ln_acled_pc ln_scad_pc pct_cov covgdp_wb flash2_tv flash2_tvgdp
 foreach i of local group{
reghdfe `i' $xlist [aw=pop_ip] , a($xlist1 $axlist cell_id)     tol(1e-02)  residuals( r_`i'3)  
}
keep pct_cov* r_* country pop_ip cell_id year gdp_g_wb adm1 adm2
save "$dir_inter/created_data_in_dta/residuals_bycell_final.dta", replace 
end

cap program drop regnow1
program define regnow1


*country X year + baseline X trend  (plus time varying vars)
global xlist=""
global xlist1 cow##year 
cap drop tmp
global axlist c.year##adm1
local group $baseline
foreach i of local group{
global xlist  $xlist c.year#`i' 
}
global xlist  $xlist $sizecontrols2 $nl $climate
regnowA


end
initial
regnow1

