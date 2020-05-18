import pandas as pd

df = pd.read_csv("venv/include/newyorkNew.csv")

#note:
#user id=1083
#category = 251

df.userId = df.userId - 1

unique_venueCategory_ids = set(df.venueCategory.values)
venueCategory2id = {}
count = 0
for category_id in unique_venueCategory_ids:
  venueCategory2id[category_id] = count
  count += 1

df['category_id'] = df.apply(lambda row: venueCategory2id[row.venueCategory], axis=1)
df.to_csv('venv/include/tokyoIdNew.csv', index=False)


