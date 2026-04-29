import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
import warnings
warnings.filterwarnings('ignore')

@st.cache_resource
def load_models():
    df = pd.read_csv('homele_v2.csv')
    df['rental_frequency'] = df['rental_frequency'].fillna('').astype(str).str.lower()
    df['is_monthly'] = df['rental_frequency'].str.contains('month', na=False).astype(int)
    df['is_yearly']  = df['rental_frequency'].str.contains('year|annual', na=False).astype(int)
    df['location_clean'] = df['location'].str.split(',').str[0].str.strip()
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
    model_sale = GradientBoostingRegressor(n_estimators=300, max_depth=4,
        learning_rate=0.05, subsample=0.8, min_samples_leaf=5, random_state=42)
    model_sale.fit(X_s, y_s)
    rent = df[df['purpose'].str.contains('Rent', na=False)].copy()
    rent = rent[(rent['price'] >= rent['price'].quantile(0.02)) &
                (rent['price'] <= rent['price'].quantile(0.98))]
    rent = rent[(rent['area'] > 20) & (rent['area'] < 2000)]
    loc_med_rent = rent.groupby('location_clean')['price'].median()
    loc_cnt_rent = rent.groupby('location_clean')['price'].count()
    loc_std_rent = rent.groupby('location_clean')['price'].std().fillna(0)
    loc_ppm2 = (rent['price']/rent['area'].replace(0,np.nan)).groupby(rent['location_clean']).median()
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
    model_rent = GradientBoostingRegressor(n_estimators=200, max_depth=3,
        learning_rate=0.05, subsample=0.7, min_samples_leaf=15,
        max_features=0.8, random_state=42)
    model_rent.fit(X_r, y_r)
    locations = sorted(df['location_clean'].dropna().unique())
    return (model_sale, model_rent, loc_med_sale, loc_med_rent,
            loc_cnt_sale, loc_cnt_rent, loc_std_rent, loc_ppm2, locations)

(model_sale, model_rent, loc_med_sale, loc_med_rent,
 loc_cnt_sale, loc_cnt_rent, loc_std_rent, loc_ppm2, locations) = load_models()

st.set_page_config(page_title="نرخی خانوو - هەرێم", page_icon="🏠")
st.title("🏠 پێشبینیکردنی نرخی خانوو - هەرێم")

tab1, tab2 = st.tabs(["پێشبینیکردن", "زانیاری مۆدێل"])
with tab1:
    purpose  = st.selectbox("بۆچی؟", ["فرۆشتن", "کرێ"])
    location = st.selectbox("ناوچە", locations)
    area     = st.number_input("مەتری خانوو", min_value=30, max_value=1000, value=150)
    beds     = st.slider("ژووری خەو", 0, 6, 2)
    baths    = st.slider("حەمام", 1, 5, 2)
    is_neg   = st.checkbox("مامەڵە هەیە ؟")

    if st.button("نرخ پێشبینی بکە", type="primary"):
        loc = str(location).strip()
        if "فرۆشتن" in purpose:
            lmp = loc_med_sale.get(loc, loc_med_sale.median())
            lc  = loc_cnt_sale.get(loc, 1)
            feat = pd.DataFrame([{
                'area':area,'bedrooms':beds,'bathrooms':baths,
                'is_negotiable':int(is_neg),'is_hot':0,'is_verified':0,
                'is_off_plan':0,'dp_ratio':0,'is_monthly':0,'is_yearly':0,
                'has_agency':0,'location_enc':0,
                'location_count':lc,'location_median_price':lmp}])
            pred   = np.expm1(model_sale.predict(feat)[0])
            margin = pred * 0.08
            st.success("نرخی پێشبینیکراو — فرۆشتن")
            st.metric("نرخ", f"${pred:,.0f}", delta=f"${margin:,.0f}")
            st.info(f"مەودا: ${pred-margin:,.0f} — ${pred+margin:,.0f}")
            if area > 500:
                st.warning("ئاگادارکردنەوە: خانووی 500m2+ کەمتر دڵنیابەخشە")
            if pred > 500000:
                st.warning("ئاگادارکردنەوە: خانووی لوکس $500k+ کەمتر دڵنیابەخشە")
        else:
            lmp = loc_med_rent.get(loc, loc_med_rent.median())
            lc  = loc_cnt_rent.get(loc, 1)
            ls  = loc_std_rent.get(loc, 0)
            lpm = loc_ppm2.get(loc, lmp/area if area > 0 else 0)
            lcv = ls/lmp if lmp > 0 else 0
            feat = pd.DataFrame([{
                'area':area,'bedrooms':beds,'bathrooms':baths,
                'is_negotiable':int(is_neg),'is_hot':0,'is_verified':0,
                'is_monthly':1,'is_yearly':0,
                'location_median_price':lmp,'location_price_std':ls,
                'location_cv':lcv,'location_count':lc,
                'loc_median_per_m2':lpm,'area_x_loc':area*lmp/1e6,
                'negotiable_vs_loc':0}])
            pred   = np.expm1(model_rent.predict(feat)[0])
            margin = pred * 0.10
            st.success("کرێی مانگانەی پێشبینیکراو")
            st.metric("مانگانە", f"${pred:,.0f}", delta=f"${margin:,.0f}")
            st.info(f"مەودا: ${pred-margin:,.0f} — ${pred+margin:,.0f}")
            st.metric("ساڵانە", f"${pred*12:,.0f}")
            if pred < 300:
                st.warning("ئاگادارکردنەوە: کرێی کەمتر لە $300 کەمتر دڵنیابەخشە")
        if lc < 5:
            st.warning(f"ئاگادارکردنەوە: {loc} تەنها {lc} خانووی هەیە")

with tab2:
    st.subheader("ئەنجامی مۆدێلەکان")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### مۆدێلی فرۆشتن")
        st.metric("CV R2", "0.913")
        st.metric("MAE", "$24,245")
        st.metric("Bias گشتی", "-0.7%")
        st.markdown("کێشەکان:")
        st.write("خانووی 500m2+ : -39%")
        st.write("خانووی $500k+ : -30%")
    with col2:
        st.markdown("### مۆدێلی کرێ")
        st.metric("CV R2", "0.773")
        st.metric("MAE", "$84/مانگ")
        st.metric("Bias گشتی", "+0.9%")
        st.markdown("کێشەکان:")
        st.write("کرێی کەمتر لە $300 : +22%")
        st.write("Star Towers : -17%")
    st.subheader("زانیاری داتا")
    st.write("سەرچاوە: homele.com - هەرێم")
    st.write("خانووی فرۆشتن: 1,288")
    st.write("خانووی کرێ: 1,129")