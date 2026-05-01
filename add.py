import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('homele_vnew.csv')
df['rental_frequency'] = df['rental_frequency'].fillna('').astype(str).str.lower()
df['is_monthly'] = df['rental_frequency'].str.contains('month', na=False).astype(int)
df['is_yearly']  = df['rental_frequency'].str.contains('year|annual', na=False).astype(int)
df['location_clean'] = df['location'].str.split(',').str[0].str.strip()

# --- SALE ---
sale = df[df['purpose'].str.contains('Sale', na=False)].copy()
sale = sale[(sale['price'] >= sale['price'].quantile(0.02)) &
            (sale['price'] <= sale['price'].quantile(0.98))]
sale = sale[(sale['area'] > 20) & (sale['area'] < 2000)]
sale['has_agency'] = 0
loc_med_sale = sale.groupby('location_clean')['price'].median()
loc_cnt_sale = sale.groupby('location_clean')['price'].count()
sale['location_median_price'] = sale['location_clean'].map(loc_med_sale)
sale['location_count']        = sale['location_clean'].map(loc_cnt_sale)
sale['location_enc']          = sale['location_clean'].astype('category').cat.codes

sale_f = ['area','bedrooms','bathrooms','is_negotiable','is_hot','is_verified',
          'is_off_plan','dp_ratio','is_monthly','is_yearly','has_agency',
          'location_enc','location_count','location_median_price']

X_s = sale[sale_f].fillna(0)
y_s = np.log1p(sale['price'])
X_tr, X_te, y_tr, y_te = train_test_split(X_s, y_s, test_size=0.2, random_state=42)

model_sale = GradientBoostingRegressor(n_estimators=300, max_depth=4,
    learning_rate=0.05, subsample=0.8, min_samples_leaf=5, random_state=42)
model_sale.fit(X_tr, y_tr)

sale_test = sale.loc[y_te.index].copy()
sale_test = sale_test.copy()
sale_test['predicted'] = np.expm1(model_sale.predict(X_te))
sale_test['actual']    = np.expm1(y_te)
sale_test['error']     = sale_test['predicted'] - sale_test['actual']
sale_test['error_pct'] = sale_test['error'] / sale_test['actual'] * 100

print("=" * 55)
print("🏠 SALE BIAS شیکاری")
print("=" * 55)

# 1. گشتی
print(f"\n1. گشتی:")
print(f"   ناوەندی هەڵە: ${sale_test['error'].mean():,.0f}")
print(f"   ئەگەر + یانی هەمیشە نرخ بەرزتر پێشبینی دەکات")
print(f"   ئەگەر - یانی هەمیشە نرخ نزمتر پێشبینی دەکات")

# 2. بەپێی مەتر
print(f"\n2. بەپێی مەتر:")
sale_test['area_group'] = pd.cut(sale_test['area'], 
    bins=[0,100,200,300,500,2000], 
    labels=['0-100','100-200','200-300','300-500','500+'])
area_bias = sale_test.groupby('area_group')['error_pct'].mean()
for grp, val in area_bias.items():
    flag = "⚠️" if abs(val) > 15 else "✅"
    print(f"   {grp}m²: {val:+.1f}%  {flag}")

# 3. بەپێی ناوچە (Top 10)
print(f"\n3. بەپێی ناوچە (Top 10 پڕ داتا):")
loc_bias = sale_test.groupby('location_clean').agg(
    count=('actual','count'),
    mean_error_pct=('error_pct','mean')
).query('count >= 5').sort_values('mean_error_pct')

print("  زیادپێشبینی (نرخ بەرزتر دەڵێت):")
for loc, row in loc_bias.tail(5).iterrows():
    print(f"   {loc:<25} {row['mean_error_pct']:+.1f}%  ({row['count']} خانوو)")

print("  کەمپێشبینی (نرخ نزمتر دەڵێت):")
for loc, row in loc_bias.head(5).iterrows():
    print(f"   {loc:<25} {row['mean_error_pct']:+.1f}%  ({row['count']} خانوو)")

# 4. بەپێی نرخ
print(f"\n4. بەپێی ئاستی نرخ:")
sale_test['price_group'] = pd.cut(sale_test['actual'],
    bins=[0,50000,100000,200000,500000,9999999],
    labels=['<50k','50-100k','100-200k','200-500k','500k+'])
price_bias = sale_test.groupby('price_group')['error_pct'].mean()
for grp, val in price_bias.items():
    flag = "⚠️" if abs(val) > 15 else "✅"
    print(f"   ${grp}: {val:+.1f}%  {flag}")

print("\n" + "=" * 55)
print("📊 کورتە:")
overall = sale_test['error_pct'].mean()
if abs(overall) < 5:
    print(f"✅ Bias کەمە ({overall:+.1f}%) — مۆدێل بەلانسە")
elif overall > 5:
    print(f"⚠️ نرخ بەرزتر دەڵێت ({overall:+.1f}%) — over-estimation")
else:
    print(f"⚠️ نرخ نزمتر دەڵێت ({overall:+.1f}%) — under-estimation")

    # --- RENT ---
rent = df[df['purpose'].str.contains('Rent', na=False)].copy()
rent = rent[(rent['price'] >= rent['price'].quantile(0.02)) &
            (rent['price'] <= rent['price'].quantile(0.98))]
rent = rent[(rent['area'] > 20) & (rent['area'] < 2000)]

loc_med_rent = rent.groupby('location_clean')['price'].median()
loc_cnt_rent = rent.groupby('location_clean')['price'].count()
loc_std_rent = rent.groupby('location_clean')['price'].std().fillna(0)
loc_ppm2     = (rent['price']/rent['area'].replace(0,np.nan)).groupby(rent['location_clean']).median()

rent['location_median_price'] = rent['location_clean'].map(loc_med_rent)
rent['location_count']        = rent['location_clean'].map(loc_cnt_rent)
rent['location_price_std']    = rent['location_clean'].map(loc_std_rent)
rent['location_cv']           = rent['location_price_std'] / rent['location_median_price'].replace(0,np.nan)
rent['loc_median_per_m2']     = rent['location_clean'].map(loc_ppm2)
rent['area_x_loc']            = rent['area'] * rent['location_median_price'] / 1e6
rent['negotiable_vs_loc']     = 0

rent_f = ['area','bedrooms','bathrooms','is_negotiable','is_hot','is_verified',
          'is_monthly','is_yearly','location_median_price','location_price_std',
          'location_cv','location_count','loc_median_per_m2','area_x_loc','negotiable_vs_loc']

X_r = rent[rent_f].fillna(0)
y_r = np.log1p(rent['price'])
X_tr, X_te, y_tr, y_te = train_test_split(X_r, y_r, test_size=0.2, random_state=42)

model_rent = GradientBoostingRegressor(n_estimators=200, max_depth=3,
    learning_rate=0.05, subsample=0.7, min_samples_leaf=15,
    max_features=0.8, random_state=42)
model_rent.fit(X_tr, y_tr)

rent_test = rent.loc[y_te.index].copy()
rent_test['predicted'] = np.expm1(model_rent.predict(X_te))
rent_test['actual']    = np.expm1(y_te)
rent_test['error']     = rent_test['predicted'] - rent_test['actual']
rent_test['error_pct'] = rent_test['error'] / rent_test['actual'] * 100

print("=" * 55)
print("🏘️  RENT BIAS شیکاری")
print("=" * 55)

# 1. گشتی
print(f"\n1. گشتی:")
print(f"   ناوەندی هەڵە: ${rent_test['error'].mean():,.0f}/مانگ")

# 2. بەپێی مەتر
print(f"\n2. بەپێی مەتر:")
rent_test['area_group'] = pd.cut(rent_test['area'],
    bins=[0,100,200,300,500,2000],
    labels=['0-100','100-200','200-300','300-500','500+'])
area_bias = rent_test.groupby('area_group')['error_pct'].mean()
for grp, val in area_bias.items():
    flag = "⚠️" if abs(val) > 15 else "✅"
    print(f"   {grp}m²: {val:+.1f}%  {flag}")

# 3. بەپێی ناوچە
print(f"\n3. بەپێی ناوچە:")
loc_bias = rent_test.groupby('location_clean').agg(
    count=('actual','count'),
    mean_error_pct=('error_pct','mean')
).query('count >= 5').sort_values('mean_error_pct')

print("  زیادپێشبینی:")
for loc, row in loc_bias.tail(5).iterrows():
    print(f"   {loc:<25} {row['mean_error_pct']:+.1f}%  ({row['count']} خانوو)")

print("  کەمپێشبینی:")
for loc, row in loc_bias.head(5).iterrows():
    print(f"   {loc:<25} {row['mean_error_pct']:+.1f}%  ({row['count']} خانوو)")

# 4. بەپێی نرخ
print(f"\n4. بەپێی ئاستی نرخ:")
rent_test['price_group'] = pd.cut(rent_test['actual'],
    bins=[0,300,600,1000,2000,99999],
    labels=['<$300','$300-600','$600-1k','$1k-2k','$2k+'])
price_bias = rent_test.groupby('price_group')['error_pct'].mean()
for grp, val in price_bias.items():
    flag = "⚠️" if abs(val) > 15 else "✅"
    print(f"   {grp}: {val:+.1f}%  {flag}")

# 5. کورتە
print(f"\n{'='*55}")
overall = rent_test['error_pct'].mean()
if abs(overall) < 5:
    print(f"✅ Bias کەمە ({overall:+.1f}%) — مۆدێل بەلانسە")
elif overall > 5:
    print(f"⚠️ نرخ بەرزتر دەڵێت ({overall:+.1f}%) — over-estimation")
else:
    print(f"⚠️ نرخ نزمتر دەڵێت ({overall:+.1f}%) — under-estimation")
