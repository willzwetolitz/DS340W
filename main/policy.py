import functions
import pandas as pd
AAPL_abs = pd.read_csv("AAPL_abs.csv")
AAPL_abs['diff'] = AAPL_abs['Mod_pred'].sub(AAPL_abs['True'].shift(1))
print(functions.basic_strategy(AAPL_abs, 100000))
print(functions.double_strategy(AAPL_abs, 100000))
print(functions.proportional_strategy(AAPL_abs, 100000, 0.3, 5))


JPM_abs = pd.read_csv("JPM_abs.csv")
JPM_abs['diff'] = JPM_abs['Mod_pred'].sub(JPM_abs['True'].shift(1))
print(functions.basic_strategy(JPM_abs, 100000))
print(functions.double_strategy(JPM_abs, 100000))
print(functions.proportional_strategy(JPM_abs, 100000, 0.3, 5))