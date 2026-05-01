import streamlit as st

st.set_page_config(
    page_title="نرخی خانوو - هەرێم",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
import warnings
warnings.filterwarnings('ignore')

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@300;400;500;700;900&display=swap');
    * { font-family: 'Tajawal', sans-serif !important; }
    .main { background: linear-gradient(135deg, #f5f7fa 0%, #e8ecf1 100%); }

    .hero-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 40px 30px; border-radius: 20px; text-align: center;
        margin-bottom: 30px; box-shadow: 0 20px 60px rgba(15,52,96,0.3);
    }
    .hero-header h1 { color: #ffffff !important; font-size: 2.5rem !important; font-weight: 900 !important; margin: 0 !important; }
    .hero-header p  { color: #a8c8f0 !important; font-size: 1rem; margin-top: 8px; }

    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #ffffff, #f8faff);
        border: 2px solid #e8f0fe; border-radius: 16px; padding: 20px !important;
        box-shadow: 0 4px 15px rgba(66,133,244,0.1); transition: transform 0.2s;
    }
    [data-testid="metric-container"]:hover { transform: translateY(-2px); }
    [data-testid="stMetricValue"] { font-size: 1.8rem !important; font-weight: 900 !important; color: #1a1a2e !important; }
    [data-testid="stMetricLabel"] { font-size: 0.9rem !important; color: #666 !important; }

    .stButton > button {
        background: linear-gradient(135deg, #0f3460, #1a6bb5) !important;
        color: white !important; border: none !important; border-radius: 12px !important;
        padding: 14px 30px !important; font-size: 1.1rem !important; font-weight: 700 !important;
        width: 100% !important; box-shadow: 0 4px 15px rgba(15,52,96,0.4) !important; transition: all 0.3s !important;
    }
    .stButton > button:hover { transform: translateY(-2px) !important; }

    .result-card {
        background: linear-gradient(135deg, #0f3460, #1a6bb5);
        border-radius: 20px; padding: 30px; color: white; text-align: center;
        margin: 20px 0; box-shadow: 0 15px 40px rgba(15,52,96,0.35);
    }
    .result-card h2 { color: white !important; font-size: 2.5rem !important; font-weight: 900 !important; margin: 10px 0 !important; }
    .result-card p  { color: #a8c8f0 !important; font-size: 1rem; }

    .fallback-box {
        background: #fff8e1; border: 2px solid #ffc107; border-radius: 12px;
        padding: 12px 16px; margin-top: 10px; font-size: 0.9rem; color: #7d5a00;
    }
    .info-box {
        background: #f8faff; border-radius: 12px; padding: 15px;
        margin-top: 15px; border: 1px solid #e0e7ff; line-height: 2;
    }
    .section-header {
        color: #1a1a2e !important; font-size: 1.2rem !important; font-weight: 700 !important;
        border-right: 4px solid #0f3460; padding-right: 12px; margin-bottom: 15px !important; display: block;
    }
    .stTabs [data-baseweb="tab-list"] {
        background: white; border-radius: 12px; padding: 4px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06); gap: 4px;
    }
    .stTabs [data-baseweb="tab"]        { border-radius: 8px !important; font-weight: 600 !important; padding: 10px 20px !important; }
    .stTabs [aria-selected="true"]      { background: linear-gradient(135deg, #0f3460, #1a6bb5) !important; color: white !important; }
    .footer { text-align: center; color: #999; font-size: 0.85rem; padding: 20px; margin-top: 30px; }
    hr { border: none; border-top: 2px solid #f0f4ff; margin: 25px 0; }
</style>
""", unsafe_allow_html=True)


# ================================================================
# لۆد کردنی مۆدێل
# ================================================================
@st.cache_resource
def load_models():
    df = pd.read_csv('homele_vnew.csv')
    df['location_clean'] = df['location'].str.split(',').str[0].str.strip()

    # ===== فرۆشتن — bedrooms >= 1 بۆ دەرکردنی کۆمەرسیاڵ =====
    sale = df[(df['type'] == 'sale') & (df['bedrooms'] >= 1)].copy()
    sale = sale[(sale['price'] >= sale['price'].quantile(0.02)) &
                (sale['price'] <= sale['price'].quantile(0.98))]
    sale = sale[(sale['area'] > 20) & (sale['area'] < 2000)]

    loc_med_sale  = sale.groupby('location_clean')['price'].median()
    loc_cnt_sale  = sale.groupby('location_clean')['price'].count()
    loc_std_sale  = sale.groupby('location_clean')['price'].std().fillna(0)
    loc_ppm2_sale = (sale['price'] / sale['area'].replace(0, np.nan)).groupby(sale['location_clean']).median()

    # گلۆباڵ — بۆ fallback
    GLOBAL_MED_S  = sale['price'].median()
    GLOBAL_STD_S  = sale['price'].std()
    GLOBAL_PPM2_S = (sale['price'] / sale['area']).median()

    sale['location_median_price'] = sale['location_clean'].map(loc_med_sale).fillna(GLOBAL_MED_S)
    sale['location_count']        = sale['location_clean'].map(loc_cnt_sale).fillna(0)
    sale['location_price_std']    = sale['location_clean'].map(loc_std_sale).fillna(GLOBAL_STD_S)
    sale['loc_median_per_m2']     = sale['location_clean'].map(loc_ppm2_sale).fillna(GLOBAL_PPM2_S)
    sale['area_x_loc']            = sale['area'] * sale['location_median_price'] / 1e6
    sale['location_enc']          = sale['location_clean'].astype('category').cat.codes

    sale_f = ['area', 'bedrooms', 'bathrooms', 'is_negotiable', 'is_hot', 'is_verified',
              'is_off_plan', 'dp_ratio', 'has_agency', 'is_ready', 'is_furnished',
              'location_enc', 'location_count', 'location_median_price',
              'location_price_std', 'loc_median_per_m2', 'area_x_loc']

    X_s = sale.reindex(columns=sale_f).fillna(0)
    y_s = np.log1p(sale['price'])

    model_sale = GradientBoostingRegressor(
        n_estimators=400, max_depth=4, learning_rate=0.04,
        subsample=0.8, min_samples_leaf=5, random_state=42)
    model_sale.fit(X_s, y_s)

    loc_enc_map = {v: i for i, v in enumerate(sorted(df['location_clean'].dropna().unique()))}

    # ===== کرێ =====
    rent = df[df['type'] == 'rent'].copy()
    rent = rent[(rent['price'] >= rent['price'].quantile(0.02)) &
                (rent['price'] <= rent['price'].quantile(0.98))]
    rent = rent[(rent['area'] > 20) & (rent['area'] < 2000)]

    loc_med_rent  = rent.groupby('location_clean')['price'].median()
    loc_cnt_rent  = rent.groupby('location_clean')['price'].count()
    loc_std_rent  = rent.groupby('location_clean')['price'].std().fillna(0)
    loc_ppm2_rent = (rent['price'] / rent['area'].replace(0, np.nan)).groupby(rent['location_clean']).median()

    GLOBAL_MED_R  = rent['price'].median()
    GLOBAL_STD_R  = rent['price'].std()
    GLOBAL_PPM2_R = (rent['price'] / rent['area']).median()

    rent['location_median_price'] = rent['location_clean'].map(loc_med_rent).fillna(GLOBAL_MED_R)
    rent['location_count']        = rent['location_clean'].map(loc_cnt_rent).fillna(0)
    rent['location_price_std']    = rent['location_clean'].map(loc_std_rent).fillna(GLOBAL_STD_R)
    rent['location_cv']           = rent['location_price_std'] / rent['location_median_price'].replace(0, np.nan)
    rent['loc_median_per_m2']     = rent['location_clean'].map(loc_ppm2_rent).fillna(GLOBAL_PPM2_R)
    rent['area_x_loc']            = rent['area'] * rent['location_median_price'] / 1e6

    # is_negotiable دەرکرا — confounding variable:
    # مامەڵە هەیە = خانووی گەورەتر = گرانتر — پێچەوانەی مەبەست
    rent_f = ['area', 'bedrooms', 'bathrooms', 'is_hot', 'is_verified',
              'is_monthly', 'is_furnished', 'location_median_price', 'location_price_std',
              'location_cv', 'location_count', 'loc_median_per_m2', 'area_x_loc']

    X_r = rent.reindex(columns=rent_f).fillna(0)
    y_r = np.log1p(rent['price'])

    model_rent = GradientBoostingRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.05,
        subsample=0.75, min_samples_leaf=10, max_features=0.8, random_state=42)
    model_rent.fit(X_r, y_r)

    locations = sorted(df['location_clean'].dropna().unique())

    return (model_sale, model_rent,
            loc_med_sale, loc_med_rent,
            loc_cnt_sale, loc_cnt_rent,
            loc_std_sale, loc_std_rent,
            loc_ppm2_sale, loc_ppm2_rent,
            GLOBAL_MED_S, GLOBAL_STD_S, GLOBAL_PPM2_S,
            GLOBAL_MED_R, GLOBAL_STD_R, GLOBAL_PPM2_R,
            loc_enc_map, locations, df)


(model_sale, model_rent,
 loc_med_sale, loc_med_rent,
 loc_cnt_sale, loc_cnt_rent,
 loc_std_sale, loc_std_rent,
 loc_ppm2_sale, loc_ppm2_rent,
 GLOBAL_MED_S, GLOBAL_STD_S, GLOBAL_PPM2_S,
 GLOBAL_MED_R, GLOBAL_STD_R, GLOBAL_PPM2_R,
 loc_enc_map, locations, df) = load_models()

MIN_LOC_COUNT = 5  # ئەگەر کەمتر بوو fallback بەکاردێت


# ================================================================
# هێدەر
# ================================================================
st.markdown("""
<div class="hero-header">
    <h1>🏠 پێشبینیکردنی نرخی خانوو</h1>
    <p>هەرێمی کوردستان — هەولێر &nbsp;|&nbsp; داتا: homele.com &nbsp;|&nbsp; 4,400+ خانوو</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🔍  پێشبینیکردن", "📊  زانیاری مۆدێل"])


# ================================================================
# TAB 1 — پێشبینیکردن
# ================================================================
with tab1:
    col_form, col_result = st.columns([1, 1], gap="large")

    with col_form:
        st.markdown('<span class="section-header">زانیاری خانووەکە</span>', unsafe_allow_html=True)

        purpose  = st.selectbox("🎯 مەبەست", ["فرۆشتن 🏷", "کرێ 🔑"])
        location = st.selectbox("📍 ناوچە", locations)
        area     = st.number_input("📐 بەرزی خانوو (م²)", min_value=30, max_value=1000, value=150, step=10)

        col_a, col_b = st.columns(2)
        with col_a:
            beds  = st.slider("🛏 ژووری خەو", 1, 6, 2)
        with col_b:
            baths = st.slider("🚿 حەمام", 1, 5, 2)

        st.markdown("---")
        st.markdown('<span class="section-header">تایبەتمەندییەکان</span>', unsafe_allow_html=True)

        col_c, col_d = st.columns(2)
        with col_c:
            is_neg       = st.checkbox("🤝 مامەڵە هەیە؟")
            is_furnished = st.checkbox("🛋️ ئامێردار؟")
        with col_d:
            is_ready    = st.checkbox("✅ ئامادەیە؟")
            is_off_plan = st.checkbox("🏗️ ئۆف پلان؟")

        # ئاگادارکردنەوە بۆ کرێ
        if "کرێ" in purpose:
            st.markdown("""
            <div style="background:#e8f4fd; border:1px solid #90caf9; border-radius:10px;
                        padding:10px 14px; font-size:0.88rem; color:#1a5276; margin-top:5px;">
                ℹ️ <b>تێبینی:</b> مامەڵە هەیە/نییە کاریگەری لەسەر مۆدێلی کرێ نییە —
                چونکە لە داتاکاندا خانووی مامەڵەدار زیاتر گەورەن نە هەرزانتر.
                نرخی پێشبینیکراو بەپێی بەرزی و ناوچە دەردێت.
            </div>
            """, unsafe_allow_html=True)

        st.markdown("")
        predict_btn = st.button("💰 نرخ پێشبینی بکە", use_container_width=True)

    # ===== ئەنجام =====
    with col_result:
        st.markdown('<span class="section-header">ئەنجامی پێشبینیکردن</span>', unsafe_allow_html=True)

        if predict_btn:
            loc = str(location).strip()

            with st.spinner("مۆدێل کار دەکات..."):

                if "فرۆشتن" in purpose:
                    # fallback logic
                    cnt = loc_cnt_sale.get(loc, 0)
                    is_fallback = cnt < MIN_LOC_COUNT

                    if not is_fallback:
                        lmp  = loc_med_sale[loc]
                        lstd = loc_std_sale.get(loc, 0)
                        lpm  = loc_ppm2_sale.get(loc, GLOBAL_PPM2_S)
                    else:
                        lmp  = GLOBAL_MED_S
                        lstd = GLOBAL_STD_S
                        lpm  = GLOBAL_PPM2_S
                        cnt  = 0

                    lenc = loc_enc_map.get(loc, 0)

                    feat = pd.DataFrame([{
                        'area': area, 'bedrooms': beds, 'bathrooms': baths,
                        'is_negotiable': int(is_neg), 'is_hot': 0, 'is_verified': 0,
                        'is_off_plan': int(is_off_plan), 'dp_ratio': 0,
                        'has_agency': 1, 'is_ready': int(is_ready),
                        'is_furnished': int(is_furnished),
                        'location_enc': lenc, 'location_count': cnt,
                        'location_median_price': lmp, 'location_price_std': lstd,
                        'loc_median_per_m2': lpm, 'area_x_loc': area * lmp / 1e6
                    }])

                    pred   = np.expm1(model_sale.predict(feat)[0])
                    margin = pred * 0.08
                    ppm2   = pred / area

                    st.markdown(f"""
                    <div class="result-card">
                        <p>✅ نرخی پێشبینیکراو — فرۆشتن</p>
                        <h2>${pred:,.0f}</h2>
                        <p>{'🛋️ ئامێردار' if is_furnished else '🏠 بێ ئامێر'}
                        &nbsp;|&nbsp; {'✅ ئامادە' if is_ready else '🏗️ نائامادە'}
                        &nbsp;|&nbsp; 📍 {loc}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    c1, c2 = st.columns(2)
                    c1.metric("📉 کەمترین", f"${pred - margin:,.0f}")
                    c2.metric("📈 زیاترین", f"${pred + margin:,.0f}")

                    st.markdown(f"""
                    <div class="info-box">
                        <b>📋 پوختە</b><br>
                        🏠 بەرزی: <b>{area} m²</b> &nbsp;|&nbsp; 💵 هەر م²: <b>${ppm2:,.0f}</b><br>
                        🛏 خەو: <b>{beds}</b> &nbsp;|&nbsp; 🚿 حەمام: <b>{baths}</b><br>
                        {'🤝 مامەڵەپێکراو' if is_neg else '💵 نرخی دیاریکراو'}
                        &nbsp;|&nbsp; {'🛋️ ئامێردار' if is_furnished else '🏠 بێ ئامێر'}
                        &nbsp;|&nbsp; {'✅ ئامادە' if is_ready else '🏗️ ئۆف پلان' if is_off_plan else '⏳ نائامادە'}
                    </div>
                    """, unsafe_allow_html=True)

                    # fallback ئاگادارکردنەوە
                    if is_fallback:
                        st.markdown(f"""
                        <div class="fallback-box">
                            ⚠️ <b>{loc}</b> داتای کافی نییە (کەمتر لە {MIN_LOC_COUNT} خانوو).<br>
                            پێشبینیکردن بەپێی ناوەندی گشتی هەرێم کراوە — کەمتر دڵنیابەخشە.
                        </div>
                        """, unsafe_allow_html=True)

                    if pred > 500000:
                        st.warning("⚠️ خانووی لوکس $500k+ — دووری -11% هەیە")
                    if pred > 300000:
                        st.warning("⚠️ خانووی $300-500k — دووری -3.7% هەیە")

                else:
                    cnt = loc_cnt_rent.get(loc, 0)
                    is_fallback = cnt < MIN_LOC_COUNT

                    if not is_fallback:
                        lmp = loc_med_rent[loc]
                        ls  = loc_std_rent.get(loc, 0)
                        lpm = loc_ppm2_rent.get(loc, GLOBAL_PPM2_R)
                        lcv = ls / lmp if lmp > 0 else 0
                    else:
                        lmp = GLOBAL_MED_R
                        ls  = GLOBAL_STD_R
                        lpm = GLOBAL_PPM2_R
                        lcv = ls / lmp if lmp > 0 else 0
                        cnt = 0

                    feat = pd.DataFrame([{
                        'area': area, 'bedrooms': beds, 'bathrooms': baths,
                        'is_hot': 0, 'is_verified': 0,
                        'is_monthly': 1, 'is_furnished': int(is_furnished),
                        'location_median_price': lmp, 'location_price_std': ls,
                        'location_cv': lcv, 'location_count': cnt,
                        'loc_median_per_m2': lpm, 'area_x_loc': area * lmp / 1e6
                    }])

                    pred   = np.expm1(model_rent.predict(feat)[0])
                    margin = pred * 0.10

                    st.markdown(f"""
                    <div class="result-card">
                        <p>✅ کرێی مانگانەی پێشبینیکراو</p>
                        <h2>${pred:,.0f}<span style="font-size:1rem; font-weight:400;">/مانگ</span></h2>
                        <p>{'🛋️ ئامێردار' if is_furnished else '🏠 بێ ئامێر'}
                        &nbsp;|&nbsp; 📍 {loc}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    c1, c2, c3 = st.columns(3)
                    c1.metric("📉 کەمترین", f"${pred - margin:,.0f}")
                    c2.metric("📈 زیاترین", f"${pred + margin:,.0f}")
                    c3.metric("📅 ساڵانە",  f"${pred * 12:,.0f}")

                    st.markdown(f"""
                    <div class="info-box">
                        <b>📋 پوختە</b><br>
                        🏠 بەرزی: <b>{area} m²</b><br>
                        🛏 خەو: <b>{beds}</b> &nbsp;|&nbsp; 🚿 حەمام: <b>{baths}</b><br>
                        {'🤝 مامەڵەپێکراو' if is_neg else '💵 نرخی دیاریکراو'}
                        &nbsp;|&nbsp; {'🛋️ ئامێردار' if is_furnished else '🏠 بێ ئامێر'}
                    </div>
                    """, unsafe_allow_html=True)

                    if is_fallback:
                        st.markdown(f"""
                        <div class="fallback-box">
                            ⚠️ <b>{loc}</b> داتای کافی نییە (کەمتر لە {MIN_LOC_COUNT} خانوو).<br>
                            پێشبینیکردن بەپێی ناوەندی گشتی هەرێم کراوە — کەمتر دڵنیابەخشە.
                        </div>
                        """, unsafe_allow_html=True)

                    if pred < 300:
                        st.warning("⚠️ کرێی کەمتر لە $300 — دووری +15% هەیە")
                    if pred > 1200:
                        st.warning("⚠️ کرێی $1200+ — دووری -17% هەیە")

        else:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #f0f4ff, #e8ecf8);
                        border-radius: 16px; padding: 50px; text-align:center;
                        border: 2px dashed #c0cef0; margin-top: 20px;">
                <div style="font-size: 3.5rem;">🏠</div>
                <p style="color: #0f3460; font-size: 1.1rem; font-weight: 700; margin-top: 15px;">
                    زانیارییەکان پڕ بکەرەوە<br>دوگمەی پێشبینیکردن داگرە
                </p>
                <p style="color: #888; font-size: 0.85rem; margin-top: 10px;">
                    فرۆشتن: CV R² = 0.950 &nbsp;|&nbsp; کرێ: CV R² = 0.811
                </p>
            </div>
            """, unsafe_allow_html=True)


# ================================================================
# TAB 2 — زانیاری مۆدێل
# ================================================================
with tab2:
    st.markdown("### 📈 بەراوردی مۆدێلەکان")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🏠 مۆدێلی فرۆشتن")
        st.metric("CV R²", "0.950", delta="↑ لە 0.913 (کۆن)")
        st.metric("MAE", "$10,359", delta="↓ لە $24,245 (کۆن)", delta_color="inverse")
        st.metric("Bias گشتی", "-0.71%")
        st.markdown("**ئاگادارییەکان:**")
        st.error("$500k+ : -11% دووری هەیە")
        st.warning("$300-500k : -3.7%")
        st.success("< $300k : باش ✅")

    with col2:
        st.markdown("#### 🏘 مۆدێلی کرێ")
        st.metric("CV R²", "0.811", delta="↑ لە 0.773 (کۆن)")
        st.metric("MAE", "$61/مانگ", delta="↓ لە $84 (کۆن)", delta_color="inverse")
        st.metric("Bias گشتی", "-1.4%")
        st.markdown("**ئاگادارییەکان:**")
        st.error("$1200+ : -17% دووری هەیە")
        st.error("< $300 : +15% دووری هەیە")
        st.success("$300-800 : باش ✅")

    st.markdown("---")
    st.markdown("### 🔧 فیکسەکانی نوێ")
    st.info("""
    **١. کۆمەرسیاڵ دەرکرا** — خانووەکانی bedrooms=0 (هۆتێل، دوکان، بینای بازرگانی) لە مۆدێلی فرۆشتن دەرکران. ئەمە CV R² بردە 0.950

    **٢. Fallback بۆ ناوچەی کەم داتا** — ئەگەر ناوچەیەک کەمتر لە 5 خانووی هەبێت (وەک 60 Meter Street، Erbil Garden)، مۆدێل بەپێی ناوەندی گشتی هەرێم پێشبینی دەکات و ئاگادارکردنەوەی نیشان دەدات.
    """)

    st.markdown("---")
    st.markdown("### 📊 نرخی ناوچەکان")

    df_chart = df.copy()
    df_chart['location_clean'] = df_chart['location'].str.split(',').str[0].str.strip()

    col_ch1, col_ch2 = st.columns(2)
    with col_ch1:
        purpose_chart = st.radio("جۆر", ["فرۆشتن", "کرێ"], horizontal=True)
    with col_ch2:
        chart_mode = st.radio(
            "شێوەی نرخ",
            ["💰 نرخی کۆتایی", "📐 نرخی هەر م²"],
            horizontal=True,
            help="نرخی هەر م² بەراوردکردنی ڕاستتری ناوچەکان دەدات"
        )

    if "فرۆشتن" in purpose_chart:
        data_c = df_chart[(df_chart['type'] == 'sale') & (df_chart['bedrooms'] >= 1)].copy()
        data_c = data_c[(data_c['price'] >= data_c['price'].quantile(0.02)) &
                        (data_c['price'] <= data_c['price'].quantile(0.98))]
        data_c = data_c[(data_c['area'] > 20) & (data_c['area'] < 2000)]
    else:
        data_c = df_chart[df_chart['type'] == 'rent'].copy()
        data_c = data_c[(data_c['price'] >= data_c['price'].quantile(0.02)) &
                        (data_c['price'] <= data_c['price'].quantile(0.98))]
        data_c = data_c[(data_c['area'] > 20) & (data_c['area'] < 2000)]

    data_c['price_per_m2'] = data_c['price'] / data_c['area']

    if "م²" in chart_mode:
        # نرخی هەر م²
        top_locs = (data_c.groupby('location_clean')
                    .agg(ppm2=('price_per_m2', 'median'), cnt=('price', 'count'))
                    .query('cnt >= 5')
                    .sort_values('ppm2', ascending=False)
                    .head(15))
        top_locs.columns = ['نرخی هەر م²', 'ژمارەی خانوو']

        st.info("📐 نرخی هەر م² — بەراوردکردنی ڕاستەقینەتر چونکە بەرزی خانوو کاریگەری نییە")
        st.bar_chart(top_locs['نرخی هەر م²'])

        # جەدوەل
        st.markdown("**15 ناوچەی بەرزترین نرخی م²:**")
        top_locs['نرخی هەر م²'] = top_locs['نرخی هەر م²'].apply(lambda x: f"${x:,.0f}")
        st.dataframe(top_locs, use_container_width=True)

    else:
        # نرخی کۆتایی
        top_locs = (data_c.groupby('location_clean')['price']
                    .agg(['median', 'count'])
                    .query('count >= 5')
                    .sort_values('median', ascending=False)
                    .head(15))
        top_locs.columns = ['نرخی ناوەند', 'ژمارەی خانوو']

        st.warning("⚠️ نرخی کۆتایی کاریگەری بەرزی خانوو تێدایە — ناوچەی خانووی گەورە دەکرێت بەرزتر دەرکەوێت")
        st.bar_chart(top_locs['نرخی ناوەند'])

        st.markdown("**15 ناوچەی بەرزترین نرخی کۆتایی:**")
        top_locs['نرخی ناوەند'] = top_locs['نرخی ناوەند'].apply(lambda x: f"${x:,.0f}")
        st.dataframe(top_locs, use_container_width=True)

    st.markdown("---")
    st.markdown("### 📦 زانیاری داتا")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("سەرچاوە", "homele.com")
    c2.metric("فرۆشتن (residential)", "2,577")
    c3.metric("کرێ", "1,140")
    c4.metric("کۆ", "4,221")

    st.markdown("---")
    st.markdown("### 🗺 مەپی ناوچەکان")

    try:
        import folium
        import json
        from streamlit_folium import st_folium

        with open('coords.json', 'r', encoding='utf-8') as f:
            coords_map = json.load(f)

        map_purpose = st.radio("جۆری مەپ", ["فرۆشتن", "کرێ"], horizontal=True, key="map_radio")
        df['location_clean'] = df['location'].str.split(',').str[0].str.strip()

        if "فرۆشتن" in map_purpose:
            price_data = df[(df['type']=='sale') & (df['bedrooms']>=1)].groupby('location_clean')['price'].median()
        else:
            price_data = df[df['type']=='rent'].groupby('location_clean')['price'].median()

        m = folium.Map(location=[36.1911, 44.0092], zoom_start=13, tiles='CartoDB positron')

        for loc_name, coord in coords_map.items():
            if coord is None:
                continue
            if not (36.0 <= coord[0] <= 36.5 and 43.8 <= coord[1] <= 44.4):
                continue

            price = price_data.get(loc_name, None)
            cnt   = df[df['location_clean'] == loc_name].shape[0]
            color = 'red' if (price and price > 200000) else 'orange' if (price and price > 100000) else 'green'

            popup_text = f"<b>{loc_name}</b><br>خانووی کۆ: {cnt}<br>"
            if price:
                popup_text += f"نرخی ناوەند: ${price:,.0f}"

            folium.CircleMarker(
                location=coord, radius=8, color=color,
                fill=True, fill_opacity=0.7,
                popup=folium.Popup(popup_text, max_width=200)
            ).add_to(m)

        st_folium(m, width=700, height=450)

    except Exception:
        st.info("🗺 مەپ بارناکرێت — folium پێویستە نصبکرێت")


# ================================================================
# فوتەر
# ================================================================
st.markdown("""
<div class="footer">
    🏠 پڕۆژەی پێشبینیکردنی نرخی خانوو — هەرێمی کوردستان<br>
    داتا: homele.com &nbsp;|&nbsp; مۆدێل: Gradient Boosting<br>
    فرۆشتن CV R² = 0.950 &nbsp;|&nbsp; کرێ CV R² = 0.811
</div>
""", unsafe_allow_html=True)
