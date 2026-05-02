# 🏠 پێشبینیکردنی نرخی خانوو — هەرێم
> **House Price Prediction — Erbil, Kurdistan Region**

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-GBR-orange)
![Status](https://img.shields.io/badge/Status-Live-brightgreen)

---

## 📌 دەربارەی پڕۆژەکە

ئەم پڕۆژەیە مۆدێلی Machine Learning بەکاردەهێنێت بۆ پێشبینیکردنی نرخی خانوو لە شاری هەرێم (ئەربیل).
داتاکان لە **homele.com** کۆکراون — مۆدێل بە **Gradient Boosting Regressor** تاقی کراوەتەوە.

🔗 **لینکی ئەپ:** [house-price-erbil1.streamlit.app](https://house-price-erbil1-nqk3mumrakybpxyyhkv6mk.streamlit.app)

---

## 📊 ئەنجامی مۆدێل

| مۆدێل | CV R² | MAE | Bias | باشبوون |
|-------|-------|-----|------|---------|
| 🏠 فرۆشتن (Sale) | **0.950** | **$10,359** | -0.71% | لە 0.913 ↑ |
| 🔑 کرێ (Rent) | **0.811** | **$61/مانگ** | -1.4% | لە 0.773 ↑ |

---

## 🗂️ فیچەرەکان

### فیچەرە بنەڕەتییەکان
| فیچەر | وەسف |
|-------|------|
| `area` | مەتری خانوو |
| `bedrooms` | ژووری خەو |
| `bathrooms` | حەمام |
| `is_furnished` | فەرنیشت/نەفەرنیشت ✅ |
| `is_off_plan` | Off-plan بوون |
| `is_ready` | ئامادەی گواستنەوە |
| `is_verified` | دابەزاندنی ڕاستەقینە |
| `is_hot` | داخوازی زیاتر |
| `has_agency` | ئەژانسی هەیە |
| `dp_ratio` | ڕێژەی دانی پێشکەش |

### فیچەرەی ناوچە
| فیچەر | وەسف |
|-------|------|
| `location_median_price` | نرخی ناوەندی ناوچەکە |
| `location_count` | ژمارەی خانووی ناوچەکە |
| `location_price_std` | جیاوازی نرخ لە ناوچەکە |
| `loc_median_per_m2` | نرخی ناوەند بە مەتر |
| `area_x_loc` | تێکەڵی مەتر و ناوچە |

---

## 🏗️ ساختەی پڕۆژەکە

```
House-Price-Erbil1/
├── model_training.py     # کۆدی سەرەکی ئەپ + مۆدێل
├── homele_vnew.csv       # داتای خانووەکان (4,398 خانوو)
├── coords2.json          # کوردینەیتی ناوچەکان (173 ناوچە)
├── requirements.txt      # پاکێجە پێویستەکان
└── README.md             # ئەم فایلە
```

---

## 📦 پاکێجە پێویستەکان

```txt
streamlit
pandas
numpy
scikit-learn
geopy
folium
streamlit-folium
```

---

## 🚀 چۆن بەکاربهێنیت (لوکاڵ)

```bash
# ١ — کلۆن بکە
git clone https://github.com/shaqlawashaqlawa33-web/House-Price-Erbil1.git
cd House-Price-Erbil1

# ٢ — پاکێجەکان دامەزرێنە
pip install -r requirements.txt

# ٣ — ئەپەکە بکەرەوە
streamlit run model_training.py
```

---

## 🔢 داتا

| | ژمارە |
|--|--|
| کۆی خانووەکان | **4,398** |
| فرۆشتن | **3,226** |
| کرێ | **1,172** |
| ناوچەکان | **284** |
| فەرنیشت | **500** |

**سەرچاوە:** [homele.com](https://homele.com) — هەرێم، ئەربیل
**کاتی کۆکردنەوە:** ٢٠٢٦

---

## 🗺️ تایبەتمەندییەکانی ئەپ

- **🔍 پێشبینیکردن** — نرخ بە مەودا پێشبینی دەکات
- **📊 چارتی ناوچەکان** — نرخی ناوەندی هەر ناوچەیەک بەراورد
- **🗺️ مەپی ئینتەراکتیڤ** — ناوچەکان لە نەخشەدا (١٧٣ ناوچە)
- **📈 زانیاری مۆدێل** — R², MAE, Bias بەراورد
- **⚠️ ئاگادارکردنەوە** — کاتێک داتای ناوچەکە کەمە

---

## ⚠️ سنوورەکان

- خانووی **500m²+** کەمتر دڵنیابەخشە
- خانووی **$500k+** کەمتر دڵنیابەخشە
- ناوچەکانی **کەمتر لە ٥ خانوو** — پێشبینی نزیکی ناوچەی دیکەی پێدەدرێت
- **Rent Bias = -1.4%** — کەمێک نرخ نزمتر دەڵێت

---

## 👤 دروستکار

**Ibrahem Qadir**
📧 shaqlawashaqlawa33@gmail.com
🔗 [GitHub](https://github.com/shaqlawashaqlawa33-web)

---

## 📄 مۆدێل

```python
# Sale Model — CV R² = 0.950
GradientBoostingRegressor(
    n_estimators=400, max_depth=4,
    learning_rate=0.04, subsample=0.8,
    min_samples_leaf=5
)

# Rent Model — CV R² = 0.811
GradientBoostingRegressor(
    n_estimators=300, max_depth=3,
    learning_rate=0.05, subsample=0.75,
    min_samples_leaf=10, max_features=0.8
)
```

---

*دوایین نوێکردنەوە: ئایاری ٢٠٢٦*
