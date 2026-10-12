import pandas as pd

train_df = pd.read_csv(r'c:\Users\Nithya\OneDrive\Documents\projects\Aircraft\AircraftComponents_RUL_Prediction\data\processed\train_FD001_processed.csv')

print("Original columns:", train_df.columns.tolist())
print("Duplicated:", train_df.columns[train_df.columns.duplicated()].tolist())

# Simulate standardize_columns
train_df.columns = [str(col).strip().lower().replace(" ", "_") for col in train_df.columns]
aliases = {"unit":"engine_id","engine":"engine_id","time":"cycle","time_in_cycles":"cycle",
           "op_setting_1":"setting_1","op_setting_2":"setting_2","op_setting_3":"setting_3"}
train_df = train_df.rename(columns=aliases)
train_df = train_df.loc[:, ~train_df.columns.astype(str).str.startswith("unnamed:")]

print("After standardize columns:", train_df.columns.tolist())
print("Duplicated after:", train_df.columns[train_df.columns.duplicated()].tolist())
print("Type of train_df['rul']:", type(train_df["rul"]))
