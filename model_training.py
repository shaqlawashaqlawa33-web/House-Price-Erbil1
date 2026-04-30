import warnings
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
import pandas as pd
import streamlit as st

# ===== ئەمە حەتمی یەکەم بێت =====
st.set_page_config(
    page_title="نرخی خانوو - هەرێم",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

warnings.filterwarnings('ignore')

# ===== ستایلی باشتر =====
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@300;400;500;700;900&display=swap');

    * { font-family: 'Tajawal', sans-serif !important; }

    .main { background: linear-gradient(135deg, #f5f7fa 0%, #e8ecf1 100%); }

    /* هێدەر */
    .hero-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 40px 30px;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 20px 60px rgba(15, 52, 96, 0.3);
    }
    .hero-header h1 {
        color: #ffffff !important;
        font-size: 2.5rem !important;
        font-weight: 900 !important;
        margin: 0 !important;
        text-shadow: 0 2px 10px rgba(0,0,0,0.3);
    }
    .hero-header p {
        color: #a8c8f0 !important;
        font-size: 1rem;
        margin-top: 8px;
    }

    /* کارتەکان */
    .card {
        background: white;
        border-radius: 16px;
        padding: 25px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        margin-bottom: 20px;
        border: 1px solid rgba(255,255,255,0.8);
    }

    /* مێتریکەکان */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #ffffff 0%, #f8faff 100%);
        border: 2px solid #e8f0fe;
        border-radius: 16px;
        padding: 20px !important;
        box-shadow: 0 4px 15px rgba(66, 133, 244, 0.1);
        transition: transform 0.2s;
    }
    [data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(66, 133, 244, 0.2);
    }
    [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 900 !important;
        color: #1a1a2e !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.9rem !important;
        color: #666 !important;
        font-weight: 500 !important;
    }

    /* دوگمە */
    .stButton > button {
        background: linear-gradient(135deg, #0f3460 0%, #1a6bb5 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 14px 30px !important;
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        width: 100% !important;
        cursor: pointer !important;
        box-shadow: 0 4px 15px rgba(15, 52, 96, 0.4) !important;
        transition: all 0.3s !important;
        letter-spacing: 0.5px;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(15, 52, 96, 0.5) !important;
    }

    /* ئینپوتەکان */
    .stSelectbox > div > div,
    .stNumberInput > div > div > input {
        border-radius: 10px !important;
        border: 2px solid #e0e7ff !important;
        background: #fafbff !important;
    }
    .stSelectbox > div > div:focus-within,
    .stNumberInput > div > div > input:focus {
        border-color: #0f3460 !important;
        box-shadow: 0 0 0 3px rgba(15, 52, 96, 0.1) !important;
    }

    /* تابەکان */
    .stTabs [data-baseweb="tab-list"] {
        background: white;
        border-radius: 12px;
        padding: 4px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06);
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        padding: 10px 20px !important;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #0f3460, #1a6bb5) !important;
        color: white !important;
    }

    /* ئەنجام کارت */
    .result-card {
        background: linear-gradient(135deg, #0f3460 0%, #1a6bb5 100%);
        border-radius: 20px;
        padding: 30px;
        color: white;
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 15px 40px rgba(15, 52, 96, 0.35);
    }
    .result-card h2 {
        color: white !important;
        font-size: 2.5rem !important;
        font-weight: 900 !important;
        margin: 10px 0 !important;
    }
    .result-card p {
        color: #a8c8f0 !important;
        font-size: 1rem;
    }

    /* بەر بەست */
    .info-badge {
        display: inline-block;
        background: #e8f0fe;
        color: #0f3460;
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 4px;
    }

    /* سێکشن هێدەر */
    .section-header {
        color: #1a1a2e !important;
        font-size: 1.2rem !important;
        font-weight: 700 !important;
        border-right: 4px solid #0f3460;
        padding-right: 12px;
        margin-bottom: 15px !important;
    }

    /* وارنینگ */
    .stWarning {
        border-radius: 10px !important;
        border-left: 4px solid #f39c12 !important;
    }
    .stSuccess {
        border-radius: 10px !important;
    }

    /* چارت */
    [data-testid="stBarChart"] {
        border-radius: 12px;
        overflow: hidden;
    }

    /* ئیموجی نوانس */
    .furnished-badge {
        background: linear-gradient(135deg, #27ae60, #2ecc71);
        color: white;
        border-radius: 8px;
        padding: 3px 10px;
        font-size: 0.8rem;
        font-weight: 700;
        margin-right: 8px;
    }

    /* دیواندەر */
    hr { border: none; border-top: 2px solid #f0f4ff; margin: 25px 0; }

    /* ڕیدایرێکشن */
    .stRadio > div { gap: 15px; }
    .stRadio > div > label {
        background: white;
        border: 2px solid #e0e7ff;
        border-radius: 10px;
        padding: 8px 16px;
        cursor: pointer;
        transition: all 0.2s;
        font-weight: 600;
    }
    .stRadio > div > label:hover {
        border-color: #0f3460;
        background: #f0f4ff;
    }

    /* فوتەر */
    .footer {
        text-align: center;
        color: #999;
        font-size: 0.85rem;
        padding: 20px;
        margin-top: 30px;
    }
</style>
""", unsafe_allow_html=True)


# ===== لۆد کردنی مۆدێل =====
@st.cache_resource
def load_models():
    df = pd.read_csv('homele_v2.csv')
    df['rental_frequency'] = df['rental_frequency'].fillna(
        '').astype(str).str.lower()
    df['is_monthly'] = df['rental_frequency'].str.contains(
        'month', na=False).astype(int)
    df['is_yearly'] = df['rental_frequency'].str.contains(
        'year|annual', na=False).astype(int)
    df['location_clean'] = df['location'].str.split(',').str[0].str.strip()

    # is_furnished column - ئەگەر هەبوو بەکاری بێنە، نەبوو 0 بکە
    if 'is_furnished' not in df.columns:
        df['is_furnished'] = 0
    else:
        df['is_furnished'] = df['is_furnished'].fillna(0).astype(int)

    # ===== مۆدێلی فرۆشتن =====
    sale = df[df['purpose'].str.contains('Sale', na=False)].copy()
    sale = sale[(sale['price'] >= sale['price'].quantile(0.02)) &
                (sale['price'] <= sale['price'].quantile(0.98))]
    sale = sale[(sale['area'] > 20) & (sale['area'] < 2000)]
    sale['has_agency'] = 0

    loc_med_sale = sale.groupby('location_clean')['price'].median()
    loc_cnt_sale = sale.groupby('location_clean')['price'].count()
    sale['location_median_price'] = sale['location_clean'].map(loc_med_sale)
    sale['location_count'] = sale['location_clean'].map(loc_cnt_sale)
    sale['location_enc'] = sale['location_clean'].astype('category').cat.codes

    sale_f = ['area', 'bedrooms', 'bathrooms', 'is_negotiable', 'is_hot', 'is_verified',
              'is_off_plan', 'dp_ratio', 'is_monthly', 'is_yearly', 'has_agency',
              'location_enc', 'location_count', 'location_median_price', 'is_furnished']
    X_s = sale.reindex(columns=sale_f).fillna(0)
    y_s = np.log1p(sale['price'])

    model_sale = GradientBoostingRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=5, random_state=42)
    model_sale.fit(X_s, y_s)

    # ===== مۆدێلی کرێ =====
    rent = df[df['purpose'].str.contains('Rent', na=False)].copy()
    rent = rent[(rent['price'] >= rent['price'].quantile(0.02)) &
                (rent['price'] <= rent['price'].quantile(0.98))]
    rent = rent[(rent['area'] > 20) & (rent['area'] < 2000)]

    loc_med_rent = rent.groupby('location_clean')['price'].median()
    loc_cnt_rent = rent.groupby('location_clean')['price'].count()
    loc_std_rent = rent.groupby('location_clean')['price'].std().fillna(0)
    loc_ppm2 = (rent['price'] / rent['area'].replace(0, np.nan)
                ).groupby(rent['location_clean']).median()

    rent['location_median_price'] = rent['location_clean'].map(loc_med_rent)
    rent['location_count'] = rent['location_clean'].map(loc_cnt_rent)
    rent['location_price_std'] = rent['location_clean'].map(loc_std_rent)
    rent['location_cv'] = rent['location_price_std'] / \
        rent['location_median_price'].replace(0, np.nan)
    rent['loc_median_per_m2'] = rent['location_clean'].map(loc_ppm2)
    rent['area_x_loc'] = rent['area'] * rent['location_median_price'] / 1e6
    rent['negotiable_vs_loc'] = 0

    rent_f = ['area', 'bedrooms', 'bathrooms', 'is_negotiable', 'is_hot', 'is_verified',
              'is_monthly', 'is_yearly', 'location_median_price', 'location_price_std',
              'location_cv', 'location_count', 'loc_median_per_m2', 'area_x_loc',
              'negotiable_vs_loc', 'is_furnished']
    X_r = rent.reindex(columns=rent_f).fillna(0)
    y_r = np.log1p(rent['price'])

    model_rent = GradientBoostingRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.7, min_samples_leaf=15, max_features=0.8, random_state=42)
    model_rent.fit(X_r, y_r)

    # ===== ناوچەکان ===== (BUG FIX: ئێستا لەناو return دایە)
    locations = sorted(df['location_clean'].dropna().unique())

    return (model_sale, model_rent, loc_med_sale, loc_med_rent,
            loc_cnt_sale, loc_cnt_rent, loc_std_rent, loc_ppm2, locations, df)


# لۆد بکە
(model_sale, model_rent, loc_med_sale, loc_med_rent,
 loc_cnt_sale, loc_cnt_rent, loc_std_rent, loc_ppm2, locations, df) = load_models()


# ===== هێدەری سەرەکی =====
st.markdown("""
<div class="hero-header">
    <h1>🏠 پێشبینیکردنی نرخی خانوو</h1>
    <p>هەرێمی کوردستان — هەولێر | داتای homele.com</p>
</div>
""", unsafe_allow_html=True)

# ===== تابەکان =====
tab1, tab2 = st.tabs(["🔍  پێشبینیکردن", "📊  زانیاری مۆدێل"])

# ================================================================
# TAB 1 — پێشبینیکردن
# ================================================================
with tab1:

    col_form, col_result = st.columns([1, 1], gap="large")

    with col_form:
        st.markdown('<p class="section-header">زانیاری خانووەکە</p>',
                    unsafe_allow_html=True)

        purpose = st.selectbox("🎯 مەبەست", ["فرۆشتن 🏷", "کرێ 🔑"])
        location = st.selectbox("📍 ناوچە", locations)
        area = st.number_input("📐 بەرزی خانوو (مەتری چوارگۆشە)",
                               min_value=30, max_value=1000, value=150, step=10)

        col_a, col_b = st.columns(2)
        with col_a:
            beds = st.slider("🛏 ژووری خەو", 0, 6, 2)
        with col_b:
            baths = st.slider("🚿 حەمام", 1, 5, 2)

        st.markdown("---")
        st.markdown('<p class="section-header">تایبەتمەندییەکان</p>',
                    unsafe_allow_html=True)

        col_c, col_d = st.columns(2)
        with col_c:
            is_neg = st.checkbox("🤝 مامەڵە هەیە؟")
        with col_d:
            is_furnished = st.checkbox("🛋️ ئامێردار؟")

        st.markdown("")
        predict_btn = st.button("💰 نرخ پێشبینی بکە", use_container_width=True)

    # ===== ئەنجام =====
    with col_result:
        st.markdown(
            '<p class="section-header">ئەنجامی پێشبینیکردن</p>', unsafe_allow_html=True)

        if predict_btn:
            loc = str(location).strip()

            with st.spinner("مۆدێل کار دەکات..."):

                if "فرۆشتن" in purpose:
                    lmp = loc_med_sale.get(loc, loc_med_sale.median())
                    lc = loc_cnt_sale.get(loc, 1)

                    # location_enc
                    df_temp = df.copy()
                    df_temp['location_clean'] = df_temp['location'].str.split(
                        ',').str[0].str.strip()
                    loc_enc_map = {v: i for i, v in enumerate(
                        sorted(df_temp['location_clean'].dropna().unique()))}
                    loc_enc = loc_enc_map.get(loc, 0)

                    feat = pd.DataFrame([{
                        'area': area, 'bedrooms': beds, 'bathrooms': baths,
                        'is_negotiable': int(is_neg), 'is_hot': 0, 'is_verified': 0,
                        'is_off_plan': 0, 'dp_ratio': 0, 'is_monthly': 0, 'is_yearly': 0,
                        'has_agency': 0, 'location_enc': loc_enc,
                        'location_count': lc, 'location_median_price': lmp,
                        'is_furnished': int(is_furnished)
                    }])

                    pred = np.expm1(model_sale.predict(feat)[0])
                    margin = pred * 0.08

                    # نمایشی ئەنجام
                    st.markdown(f"""
                    <div class="result-card">
                        <p>✅ نرخی پێشبینیکراو — فرۆشتن</p>
                        <h2>${pred:,.0f}</h2>
                        <p>{'🛋️ ئامێردار' if is_furnished else '🏠 بێ ئامێر'} &nbsp;|&nbsp; 📍 {loc}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    c1, c2 = st.columns(2)
                    c1.metric("📉 کەمترین نرخ", f"${pred-margin:,.0f}")
                    c2.metric("📈 زیاترین نرخ", f"${pred+margin:,.0f}")

                    # زانیاری زیاتر
                    st.markdown(f"""
                    <div style="background:#f8faff; border-radius:12px; padding:15px; margin-top:15px; border:1px solid #e0e7ff;">
                        <b>📋 پوختە</b><br><br>
                        🏠 بەرزی خانوو: <b>{area} m²</b><br>
                        🛏 ژووری خەو: <b>{beds}</b> &nbsp;|&nbsp; 🚿 حەمام: <b>{baths}</b><br>
                        {'🤝 مامەڵەپێکراو' if is_neg else '💵 نرخی دیاریکراو'}<br>
                        {'🛋️ ئامێردار' if is_furnished else '🏠 بێ ئامێر'}
                    </div>
                    """, unsafe_allow_html=True)

                    if lc < 5:
                        st.warning(
                            f"⚠️ {loc} تەنها {lc} خانووی هەیە — کەمتر دڵنیابەخشە")
                    if area > 500:
                        st.warning("⚠️ خانووی 500m²+ کەمتر دڵنیابەخشە")
                    if pred > 500000:
                        st.warning("⚠️ خانووی لوکس $500k+ کەمتر دڵنیابەخشە")

                else:
                    lmp = loc_med_rent.get(loc, loc_med_rent.median())
                    lc = loc_cnt_rent.get(loc, 1)
                    ls = loc_std_rent.get(loc, 0)
                    lpm = loc_ppm2.get(loc, lmp / area if area > 0 else 0)
                    lcv = ls / lmp if lmp > 0 else 0

                    feat = pd.DataFrame([{
                        'area': area, 'bedrooms': beds, 'bathrooms': baths,
                        'is_negotiable': int(is_neg), 'is_hot': 0, 'is_verified': 0,
                        'is_monthly': 1, 'is_yearly': 0,
                        'location_median_price': lmp, 'location_price_std': ls,
                        'location_cv': lcv, 'location_count': lc,
                        'loc_median_per_m2': lpm, 'area_x_loc': area * lmp / 1e6,
                        'negotiable_vs_loc': 0, 'is_furnished': int(is_furnished)
                    }])

                    pred = np.expm1(model_rent.predict(feat)[0])
                    margin = pred * 0.10

                    st.markdown(f"""
                    <div class="result-card">
                        <p>✅ کرێی مانگانەی پێشبینیکراو</p>
                        <h2>${pred:,.0f}<span style="font-size:1rem; font-weight:400;">/مانگ</span></h2>
                        <p>{'🛋️ ئامێردار' if is_furnished else '🏠 بێ ئامێر'} &nbsp;|&nbsp; 📍 {loc}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    c1, c2, c3 = st.columns(3)
                    c1.metric("📉 کەمترین", f"${pred-margin:,.0f}")
                    c2.metric("📈 زیاترین", f"${pred+margin:,.0f}")
                    c3.metric("📅 ساڵانە", f"${pred*12:,.0f}")

                    st.markdown(f"""
                    <div style="background:#f8faff; border-radius:12px; padding:15px; margin-top:15px; border:1px solid #e0e7ff;">
                        <b>📋 پوختە</b><br><br>
                        🏠 بەرزی خانوو: <b>{area} m²</b><br>
                        🛏 ژووری خەو: <b>{beds}</b> &nbsp;|&nbsp; 🚿 حەمام: <b>{baths}</b><br>
                        {'🤝 مامەڵەپێکراو' if is_neg else '💵 نرخی دیاریکراو'}<br>
                        {'🛋️ ئامێردار' if is_furnished else '🏠 بێ ئامێر'}
                    </div>
                    """, unsafe_allow_html=True)

                    if lc < 5:
                        st.warning(
                            f"⚠️ {loc} تەنها {lc} خانووی هەیە — کەمتر دڵنیابەخشە")
                    if pred < 300:
                        st.warning("⚠️ کرێی کەمتر لە $300 کەمتر دڵنیابەخشە")

        else:
            # پلەیسهۆڵدەر
            st.markdown("""
            <div style="background: linear-gradient(135deg, #f0f4ff, #e8ecf8);
                        border-radius: 16px; padding: 40px; text-align:center;
                        border: 2px dashed #c0cef0; margin-top: 30px;">
                <div style="font-size: 3rem;">🏠</div>
                <p style="color: #0f3460; font-size: 1.1rem; font-weight: 600; margin-top: 15px;">
                    زانیارییەکان پڕ بکەرەوە و دوگمەی پێشبینیکردن داگرە
                </p>
                <p style="color: #888; font-size: 0.9rem;">
                    مۆدێل CV R² = 0.913 بۆ فرۆشتن
                </p>
            </div>
            """, unsafe_allow_html=True)


# ================================================================
# TAB 2 — زانیاری مۆدێل
# ================================================================
with tab2:

    st.markdown("### 📈 ئەنجامی مۆدێلەکان")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="card">
            <h3 style="color:#0f3460; margin-top:0;">🏠 مۆدێلی فرۆشتن</h3>
        """, unsafe_allow_html=True)
        st.metric("CV R²", "0.913", delta="زۆر باش ✅")
        st.metric("MAE", "$24,245")
        st.metric("Bias گشتی", "-0.7%")
        st.error("خانووی 500m²+ : -39% دووری هەیە")
        st.error("خانووی $500k+ : -30% دووری هەیە")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card">
            <h3 style="color:#0f3460; margin-top:0;">🏘 مۆدێلی کرێ</h3>
        """, unsafe_allow_html=True)
        st.metric("CV R²", "0.773", delta="باش ✅")
        st.metric("MAE", "$84/مانگ")
        st.metric("Bias گشتی", "+0.9%")
        st.error("کرێی کەمتر لە $300 : +22% دووری هەیە")
        st.error("Star Towers : -17% دووری هەیە")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    # ===== چارتی ناوچەکان =====
    st.markdown("### 📊 نرخی ناوەندی بەپێی ناوچە")

    df_chart = df.copy()
    df_chart['location_clean'] = df_chart['location'].str.split(
        ',').str[0].str.strip()
    purpose_chart = st.radio("جۆر بە", ["فرۆشتن", "کرێ"], horizontal=True)

    if "فرۆشتن" in purpose_chart:
        data = df_chart[df_chart['purpose'].str.contains('Sale', na=False)]
    else:
        data = df_chart[df_chart['purpose'].str.contains('Rent', na=False)]

    top_locs = (data.groupby('location_clean')['price']
                .agg(['median', 'count'])
                .query('count >= 5')
                .sort_values('median', ascending=False)
                .head(15))
    top_locs.columns = ['نرخی ناوەند', 'ژمارەی خانوو']
    st.bar_chart(top_locs['نرخی ناوەند'])

    st.markdown("---")

    # ===== زانیاری داتا =====
    st.markdown("### 📦 زانیاری داتا")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("سەرچاوە", "homele.com")
    c2.metric("خانووی فرۆشتن", "1,288")
    c3.metric("خانووی کرێ", "1,129")
    c4.metric("کۆی گشتی", "2,417")

    st.markdown("---")

    # ===== مەپ =====
    st.markdown("### 🗺 مەپی ناوچەکان")

    try:
        import folium
        import json
        from streamlit_folium import st_folium

        with open('coords.json', 'r', encoding='utf-8') as f:
            coords_map = json.load(f)

        map_purpose = st.radio(
            "جۆری مەپ", ["فرۆشتن", "کرێ"], horizontal=True, key="map_radio")

        if "فرۆشتن" in map_purpose:
            price_data = df[df['purpose'].str.contains('Sale', na=False)].groupby(
                'location_clean')['price'].median()
        else:
            price_data = df[df['purpose'].str.contains('Rent', na=False)].groupby(
                'location_clean')['price'].median()

        m = folium.Map(location=[36.1911, 44.0092],
                       zoom_start=13, tiles='CartoDB positron')

        for loc_name, coord in coords_map.items():
            if coord is None:
                continue
            if not (36.0 <= coord[0] <= 36.5 and 43.8 <= coord[1] <= 44.4):
                continue

            price = price_data.get(loc_name, None)
            cnt = df[df['location_clean'] == loc_name].shape[0]

            if price and price > 200000:
                color = 'red'
            elif price and price > 100000:
                color = 'orange'
            else:
                color = 'green'

            popup_text = f"<b>{loc_name}</b><br>خانووی کۆ: {cnt}<br>"
            if price:
                popup_text += f"نرخی ناوەند: ${price:,.0f}"

            folium.CircleMarker(
                location=coord, radius=8, color=color,
                fill=True, fill_opacity=0.7,
                popup=folium.Popup(popup_text, max_width=200)
            ).add_to(m)

        st_folium(m, width=700, height=450)

    except Exception as e:
        st.info("🗺 مەپ بارناکرێت — folium پێویستە نصبکرێت")

# ===== فوتەر =====
st.markdown("""
<div class="footer">
    🏠 پڕۆژەی پێشبینیکردنی نرخی خانوو — هەرێمی کوردستان<br>
    داتا: homele.com &nbsp;|&nbsp; مۆدێل: Gradient Boosting &nbsp;|&nbsp; تەواوکراو ✅
</div>
""", unsafe_allow_html=True)
