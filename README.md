## Overview
This project constructs stochastic life tables using Human Mortality Database data.
We focus on uncertainty quantification of mortality rates and life expectancy.

## Data
- Human Mortality Database (France)
- Death counts and exposures by age, sex, year

## Methodology
- Poisson mortality model for death counts
- Analytical variance of mortality rates
- Delta Method for life table quantities
- Parametric bootstrap for life expectancy confidence intervals

## Outputs
- Age-specific mortality rates with confidence intervals
- Complete life tables
- Confidence intervals for life expectancy at birth

## Tools
Python, NumPy, Pandas