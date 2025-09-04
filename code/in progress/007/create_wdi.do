# delimit ;
clear all;
cap log close;
cap file close _all;
cap program drop _all;
set scrollbufsize 1000000;
set more off;


*******************************************;
* Generates database with World Bank series;
* MPM 2016-09-21;
*******************************************;


*** SETTINGS ***;

* Decide what to do;
local create_db = 1; // Create WDI DB;
local summ_miss = 1; // Summarize data and missings;

* Filenames;
local csv_file = "../../data/WDI_Data_20160921.csv"; // WDI CSV file;
local miss_file = "output/wdi_miss.csv"; // Output file for missing data calculations;

* WDI CSV spreadsheet settings;
local wdi_year_start = 1960; // First year in WDI sheet;

* Selected sample;
local year_start = 1989;
local year_end = 2015;

* Variables;
local var_keep = "FP_CPI_TOTL FP_CPI_TOTL_ZG FP_WPI_TOTL NE_EXP_GNFS_CD NE_EXP_GNFS_KD NE_IMP_GNFS_CD NE_IMP_GNFS_KD NY_GDP_MKTP_CD NY_GDP_MKTP_KD NY_GDP_DEFL_KD_ZG PA_NUS_FCRF PA_NUS_ATLS"; // WDI series to keep;

* Eurozone;
local ez = "AUT BEL DEU ESP EST FIN FRA GRC IRL ITA LTU LUX NLD PRT SVK SVN";

*** END SETTINGS ***;


*** GENERATE DB ***;

if `create_db' == 1 {;

import delimited using `csv_file', varnames(1) case(preserve) clear; // Import raw WDI data;
drop CountryName;

* Rename year variables;
local y = `wdi_year_start';
foreach s of varlist v* {;
	if `y' < `year_start' {;
		drop `s';
	};
	else {;
		rename `s' val_`y'; // Denote the value at year y by val_y;
	};
	local y = `y'+1;
};

replace IndicatorCode = subinstr(IndicatorCode, ".", "_", .); // Replace periods in variable codes with underscores;
keep if strpos("`var_keep'", IndicatorCode)>0; // Keep only desired WDI series;

preserve; // Store data in memory;

collapse (first) IndicatorCode, by(IndicatorName); // Collapse to variable codes/names;

* Store variable codes and names so we can make variable labels later;
local label_num = `=_N';
forvalues i=1/`label_num' {;
	local label_code_`i' = "`=IndicatorCode[`i']'";
	local label_name_`i' = "`=IndicatorName[`i']'";
};

restore; // Bring back original data;

drop IndicatorName;
rename CountryCode country_iso;

replace country_iso = "ROU" if country_iso == "ROM"; // Handle Romania;

merge m:1 country_iso using countries, nogen keep(match); // Merge with country list from COMTRADE, keep only matched countries;
drop country_iso;

reshape long val_, i(country IndicatorCode) j(year) string; // Reshape data set so different years are not separate variables;
destring year, replace;

reshape wide val_, i(country year) j(IndicatorCode) string; // Reshape data set so different WDI series go along columns;
rename val_* *;

format year %ty;
tsset country year; // Set panel structure;
order *, alpha;
order country year;
compress;

forvalues i=1/`label_num' {;
	label var `label_code_`i'' "`label_name_`i''";
};

label data "World Bank World Development Indicators";
save wdi, replace;

};


*** SUMMARIZE DATA ***;

if `summ_miss' == 1 {;

use wdi, clear;

su *; // Summarize data;
bys country: su *; // Summarize data by country;

foreach v of varlist `var_keep' {;
	gen `v'_first = cond(missing(`v'), ., year);
	gen `v'_last = `v'_first;
};

* Count missing data by country;
collapse (count) `var_keep' (firstnm) *_first (lastnm) *_last, by(country);

local vlist = "";
foreach v of varlist `var_keep' {;
	qui: gen `v'_range = string(`v'_first) + "-" + string(`v'_last);
	qui: replace `v'_range = "" if missing(`v'_first);
	local vlist = "`vlist' `v' `v'_range";
};
sort country;

export delimited country `vlist' using `miss_file', replace;

};

