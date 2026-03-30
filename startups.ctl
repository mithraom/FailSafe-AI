LOAD DATA
INFILE 'startups.csv'
INTO TABLE startups
FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED BY '"'
IGNORE 1 LINES
(
    Startup,
    Industry,
    MonthlyRevenue "TO_NUMBER(:MonthlyRevenue)",
    MonthlyExpenses "TO_NUMBER(:MonthlyExpenses)",
    GrowthRate "TO_NUMBER(:GrowthRate)",
    MonthsRunway,
    TeamSize,
    MarketRisk,
    Funding "TO_NUMBER(:Funding)"
)