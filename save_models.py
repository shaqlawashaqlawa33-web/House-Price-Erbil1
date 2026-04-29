import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('homele_v2.csv')

# --- پاککردنەوە ---
df['rental_frequency'] = df['rental_frequency'].fillna(
    '').astype(str).str.lower()
df['is_monthly'] = df['rental_frequency'].str.contains(
    'month', na=False).astype(int)
df['is_yearly'] = df['rental_frequency'].str.contains(
    'year|annual', na=False).astype(int)
df['location_clean'] = df['location'].str.split(',').str[0].str.strip()


# --- SALE ---
sale = df[df['purpose'].str.contains('Sale', na=False)].copy()
sale = sale[(sale['price'] >= sale['price'].quantile(0.02)) &
            (sale['price'] <= sale['price'].quantile(0.98))]
sale = sale[(sale['area'] > 20) & (sale['area'] < 2000)]

loc_med_sale = sale.groupby('location_clean')['price'].median()
loc_cnt_sale = sale.groupby('location_clean')['price'].count()

sale['location_median_price'] = sale['location_clean'].map(loc_med_sale)
sale['location_count'] = sale['location_clean'].map(loc_cnt_sale)
sale['location_enc'] = sale['location_clean'].astype('category').cat.codes

sale['has_agency'] = 0

sale_features = ['area', 'bedrooms', 'bathrooms', 'is_negotiable',
                 'is_hot', 'is_verified', 'is_off_plan', 'dp_ratio',
                 'is_monthly', 'is_yearly', 'has_agency',
                 'location_enc', 'location_count', 'location_median_price']

X_sale = sale[sale_features].fillna(0)
y_sale = np.log1p(sale['price'])
X_tr, X_te, y_tr, y_te = train_test_split(
    X_sale, y_sale, test_size=0.2, random_state=42)

model_sale = GradientBoostingRegressor(n_estimators=300, max_depth=4,
                                       learning_rate=0.05, subsample=0.8, min_samples_leaf=5, random_state=42)
model_sale.fit(X_tr, y_tr)
print(f"✅ Sale R²: {r2_score(y_te, model_sale.predict(X_te)):.4f}")

# --- RENT ---
rent = df[df['purpose'].str.contains('Rent', na=False)].copy()
rent = rent[(rent['price'] >= rent['price'].quantile(0.02)) &
            (rent['price'] <= rent['price'].quantile(0.98))]
rent = rent[(rent['area'] > 20) & (rent['area'] < 2000)]

loc_med_rent = rent.groupby('location_clean')['price'].median()
loc_cnt_rent = rent.groupby('location_clean')['price'].count()
loc_std_rent = rent.groupby('location_clean')['price'].std().fillna(0)
loc_ppm2 = (rent['price']/rent['area'].replace(0, np.nan)
            ).groupby(rent['location_clean']).median()

rent['location_median_price'] = rent['location_clean'].map(loc_med_rent)
rent['location_count'] = rent['location_clean'].map(loc_cnt_rent)
rent['location_price_std'] = rent['location_clean'].map(loc_std_rent)
rent['location_cv'] = rent['location_price_std'] / \
    rent['location_median_price'].replace(0, np.nan)
rent['loc_median_per_m2'] = rent['location_clean'].map(loc_ppm2)
rent['area_x_loc'] = rent['area'] * rent['location_median_price'] / 1e6
rent['negotiable_vs_loc'] = (rent['is_negotiable'] *
                             (rent['price'] - rent['location_median_price']) /
                             rent['location_median_price'].replace(0, np.nan)).fillna(0)

rent_features = ['area', 'bedrooms', 'bathrooms', 'is_negotiable',
                 'is_hot', 'is_verified', 'is_monthly', 'is_yearly',
                 'location_median_price', 'location_price_std',
                 'location_cv', 'location_count',
                 'loc_median_per_m2', 'area_x_loc', 'negotiable_vs_loc']

X_rent = rent[rent_features].fillna(0)
y_rent = np.log1p(rent['price'])
X_tr, X_te, y_tr, y_te = train_test_split(
    X_rent, y_rent, test_size=0.2, random_state=42)

model_rent = GradientBoostingRegressor(n_estimators=200, max_depth=3,
                                       learning_rate=0.05, subsample=0.7, min_samples_leaf=15,
                                       max_features=0.8, random_state=42)
model_rent.fit(X_tr, y_tr)
print(f"✅ Rent R²: {r2_score(y_te, model_rent.predict(X_te)):.4f}")
# --- مەحفوزکردن ---
pickle.dump(model_sale,  open('model_sale.pkl',  'wb'))
pickle.dump(model_rent,  open('model_rent.pkl',  'wb'))
pickle.dump(loc_med_sale, open('loc_med_sale.pkl', 'wb'))
pickle.dump(loc_med_rent, open('loc_med_rent.pkl', 'wb'))
pickle.dump(loc_cnt_sale, open('loc_cnt_sale.pkl', 'wb'))
pickle.dump(loc_cnt_rent, open('loc_cnt_rent.pkl', 'wb'))
pickle.dump(loc_std_rent, open('loc_std_rent.pkl', 'wb'))
pickle.dump(loc_ppm2,     open('loc_ppm2.pkl',     'wb'))
pickle.dump(sorted(df['location_clean'].dropna().unique()),
            open('locations.pkl', 'wb'))

print("✅ هەموو فایلەکان مەحفوزکران!")
