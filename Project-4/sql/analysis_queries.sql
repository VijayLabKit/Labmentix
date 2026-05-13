-- Geographical Insight: Top 10 Districts by Transaction Value (2023)
SELECT District, State, SUM(Amount) as Total_Amount
FROM map_transaction
WHERE Year = 2023
GROUP BY District, State
ORDER BY Total_Amount DESC
LIMIT 10;

-- User Insight: Brand Dominance by State
SELECT State, Brand, SUM(Count) as User_Count
FROM aggregated_user
GROUP BY State, Brand
ORDER BY State, User_Count DESC;

-- Insurance Adoption: State-wise Growth
SELECT State, Year, SUM(Amount) as Insurance_Value
FROM aggregated_insurance
GROUP BY State, Year
ORDER BY State, Year;

-- Pincode Performance: High Value Clusters
SELECT Pincode, State, SUM(Amount) as Total_Value
FROM top_transaction
GROUP BY Pincode, State
ORDER BY Total_Value DESC
LIMIT 20;