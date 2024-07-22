# EDA Notes
Running List of all takeaways and notes from EDA
* `BsmtFinSF2` has very low correlation with target. We will remove this
* `LowQualFinSF` has very low correlation with target. We will remove this
* `BsmtHalfBath` has very low correlation with target. We will remove this
* `MiscVal` has very low correlation with target. We will remove this
* Target variable `SalePrice` is a bit skewed. We could try a log transformation
* Several of the numeric variables should be treated as ordinal categoricals:
  * `OverallQual`
  * `OverallCond`
  * `YearBuilt`
  * `YearRemodAdd`
    * This needs transformation - if same as construction date, set to NA
  * `BsmtFullBath`
  * `BsmtHalfBath`
  * `FullBath`
  * `HalfBath`
  * `BedroomAbvGr`
  * `KitchenAbvGr`
  * `TotRmsAbvGrd`
  * `Fireplaces`
  * `GarageYrBlt`
  * `GarageCars`
  * `MoSold`
  * `YrSold`
* Compute ```TotalBath = BsmtFullBath + BsmtHalfBath + FullBath + HalfBath```