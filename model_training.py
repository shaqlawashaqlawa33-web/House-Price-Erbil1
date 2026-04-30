import warnings
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium

# 1. ڕێکخستنی لاپەڕە
st.set_page_config(
    page_title="نرخی خانوو - هەرێم",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

warnings.filterwarnings('ignore')

# 2. ستایلی دیزاین (CSS)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@300;400;500;700;900&display=swap');
    * { font-family: 'Tajawal', sans-serif !important; direction: rtl; }
    .main { background: linear-gradient(135deg, #f5f7fa 0%, #e8ecf1 100%); }
    .hero-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 40px; border-radius: 20px; text-align: center; color: white; margin-bottom: 30px;
    }
    .card { background: white; padding: 25px; border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.05); }
</style>
""", unsafe_allow_html=True)

# 3. بارکردن و پاککردنەوەی داتا (لۆژیکی نوێ)
@st.cache_data
def load_and_clean_all():
    df = pd.read_csv('homele_v2.csv')
    df['location_clean'] = df['location'].str.split(',').str[0].str.strip()
    
    # پاککردنەوەی داتای فرۆشتن
    sale_df = df[(df['purpose'] == 'For Sale') & (df['price'] >= 15000) & (df['price'] <= 2000000)].copy()
    
    # پاککردنەوەی داتای کرێ
    rent_df = df[(df['purpose'] == 'For Rent') & (df['price'] >= 100) & (df['price'] <= 10000)].copy()
    
    for d in [sale_df, rent_df]:
        d['bedrooms'] = d['bedrooms'].fillna(d['bedrooms'].median())
        d['bathrooms'] = d['bathrooms'].fillna(d['bathrooms'].median())
        d['area'] = d['area'].fillna(d['area'].median())
        
    return sale_df, rent_df

sale_data, rent_data = load_and_clean_all()

# 4. فەرمانی ڕاهێنانی مۆدێل
@st.cache_resource
def train_professional_models(data):
    if data.empty: return None, None
    X = pd.get_dummies(data[['area', 'bedrooms', 'bathrooms', 'location_clean']])
    y = data['price']
    model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42)
    model.fit(X, y)
    return model, X.columns

model_sale, cols_sale = train_professional_models(sale_data)
model_rent, cols_rent = train_professional_models(rent_data)

# 5. دیزاینی سەرەوە
st.markdown('<div class="hero-header"><h1>سیستەمی پێشبینیکردنی نرخی خانوو 🏠</h1><p>بە بەکارهێنانی ژیریی دەستکرد و داتاکانی هەرێم</p></div>', unsafe_allow_html=True)

# 6. تاشبۆرد و بەشەکان
tab1, tab2, tab3 = st.tabs(["💰 خەمڵاندنی فرۆشتن", "🔑 خەمڵاندنی کرێ", "📍 نەخشەی نرخەکان"])

# --- بەشی فرۆشتن ---
with tab1:
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            s_area = st.number_input("ڕووبەر (m²)", 50, 2000, 150, key="s1")
            s_loc = st.selectbox("شوێن", sorted(sale_data['location_clean'].unique()), key="s2")
        with col2:
            s_beds = st.slider("ژووری نووستن", 1, 10, 3, key="s3")
            s_baths = st.slider("گەرماو", 1, 8, 2, key="s4")
        
        if st.button("پێشبینیکردنی نرخی کڕین"):
            input_df = pd.DataFrame([[s_area, s_beds, s_baths, s_loc]], columns=['area', 'bedrooms', 'bathrooms', 'location_clean'])
            input_encoded = pd.get_dummies(input_df).reindex(columns=cols_sale, fill_value=0)
            res = model_sale.predict(input_encoded)[0]
            st.markdown(f"<h2 style='text-align:center; color:#16213e;'>نرخی خەمڵێنراو: ${res:,.0f}</h2>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# --- بەشی کرێ ---
with tab2:
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            r_area = st.number_input("ڕووبەر (m²)", 50, 1000, 120, key="r1")
            r_loc = st.selectbox("شوێن", sorted(rent_data['location_clean'].unique()), key="r2")
        with col2:
            r_beds = st.slider("ژووری نووستن", 1, 8, 2, key="r3")
            r_baths = st.slider("گەرماو", 1, 6, 1, key="r4")
            
        if st.button("پێشبینیکردنی کرێ"):
            input_df = pd.DataFrame([[r_area, r_beds, r_baths, r_loc]], columns=['area', 'bedrooms', 'bathrooms', 'location_clean'])
            input_encoded = pd.get_dummies(input_df).reindex(columns=cols_rent, fill_value=0)
            res = model_rent.predict(input_encoded)[0]
            st.markdown(f"<h2 style='text-align:center; color:#0f3460;'>کرێی مانگانە: ${res:,.0f}</h2>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# --- بەشی نەخشە ---
with tab3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("دابەشبوونی نرخەکان لەسەر نەخشە")
    
    # لێرەدا کۆردیناتەکان وەەک ئەوەی کۆدەکەتە بەڵام بۆ سەرجەم داتاکان
    coords_map = {
        "Vana Towers": [36.2345, 44.0012], "Empire Wings": [36.2089, 43.9912],
        "White Towers": [36.1822, 44.0234], "Life Towers": [36.1950, 43.9870],
        "Lavender Skyline Tower": [36.2150, 44.0300], "Tulip Towers": [36.2200, 43.9980]
    }
    
    m = folium.Map(location=[36.1911, 44.0092], zoom_start=13, tiles='CartoDB positron')
    
    # بەکارهێنانی داتای فرۆشتن بۆ نەخشەکە وەک نموونە
    price_data = sale_data.groupby('location_clean')['price'].median()
    
    for loc_name, coord in coords_map.items():
        if loc_name in price_data:
            price = price_data[loc_name]
            color = 'red' if price > 150000 else 'orange' if price > 80000 else 'green'
            folium.CircleMarker(
                location=coord, radius=10, color=color, fill=True,
                popup=f"{loc_name}: ${price:,.0f}"
            ).add_to(m)
    
    st_folium(m, width=1200, height=500)
    st.markdown('</div>', unsafe_allow_html=True)
