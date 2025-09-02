# issues:
both tables: When using wage differences as the outcome, many values are negative (since wages can decrease). The authors handle this by taking log(|wage_change|) and then multiplying by the original sign to preserve directionality: sign(x) × log(|x|). However, our replicate function expects positive level values to log internally and cannot implement this signed-log transformation.

# Meet Comment:
Take their variable and just exponentiate it. Though not sure about what they are intending to do with that var.