# issues:
both tables: When using wage differences as the outcome, many values are negative (since wages can decrease). The authors handle this by taking log(|wage_change|) and then multiplying by the original sign to preserve directionality: sign(x) × log(|x|). However, our replicate function expects positive level values to log internally and cannot implement this signed-log transformation.

# Meet Comment:
Take their variable and exponentiate, should be replicable - if doesn't run contact me.