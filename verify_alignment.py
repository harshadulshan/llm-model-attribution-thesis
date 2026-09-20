import pandas as pd
df = pd.read_csv("features_extracted.csv")
orig = df[df['edit_condition']=='original'].reset_index(drop=True)
light = df[df['edit_condition']=='light'].reset_index(drop=True)
heavy = df[df['edit_condition']=='heavy'].reset_index(drop=True)

mismatches = 0
for i in range(len(orig)):
    if not (orig.loc[i,'row_position'] == light.loc[i,'row_position'] == heavy.loc[i,'row_position']):
        mismatches += 1

print(f"Total samples: {len(orig)}")
print(f"Row alignment mismatches: {mismatches}")
print("ALIGNED CORRECTLY" if mismatches == 0 else "MISALIGNED - RESULTS ARE INVALID")
