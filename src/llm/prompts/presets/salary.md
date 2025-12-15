**Domain Specific Instructions (Salary Forecasting):**

1.  **Targets**:
    - Look for columns representing financial compensation (e.g. total yearly compensation, base salary, bonuses, stock grants).

2.  **Levels**:
    - Identify columns representing job hierarchy or seniority.
    - Rank them semantically from junior to senior roles.

3.  **Locations**:
    - Identify geographical columns (City, Metro, Office Location).
    - Map them to Cost Tiers based on general tech industry cost-of-living standards (e.g. Major Tech Hubs vs Remote/Low COL).

4.  **Constraints**:
    - Apply positive monotonicity constraints to features that naturally correlate with higher compensation, such as experience, tenure, or job level.
