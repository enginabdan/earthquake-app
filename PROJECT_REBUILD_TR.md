# Planets Projesi - Yeniden Kurulum ve Bilimsel Yol Haritasi

Bu repo su an SPICE ogrenme notlari seviyesinde. Hedefin olan analiz icin veri hatti ve modelleme katmani bu dokumanla yeniden tanimlandi.

## 1) Hedefin Teknik Karsiligi

Hedef: `2000-01-01` tarihinden itibaren dakika bazinda gezegen konumlarini ve `M>=4` depremleri ayni zaman ekseninde birlestirip, istatistiksel olarak anlamli bir iliski var mi test etmek.

Onemli not: Bu calisma nedensellik ispatlamaz. En fazla, veri icinde tekrar eden bir korelasyon olup olmadigini test eder.

## 2) Veri Kaynaklari

1. Deprem verisi: USGS Earthquake Catalog API
2. Gezegen efemeris: NASA NAIF SPICE kernel (`de405.bsp` + `naif0012.tls`)

Neden bu secim:
- Dakika seviyesinde zaman damgasi ile tutarli, tekrar uretilebilir ve anahtarsiz (USGS + SPICE lokal) calisir.
- Su an `planet.ipynb` icindeki Timeanddate API anahtarli akis, bu arastirma icin gerekli degil.

## 3) Bu Repoya Eklenen Yeni Pipeline

`pipeline/` altinda:

1. `fetch_earthquakes.py`
- 2000'den itibaren aylik pencerelerle USGS'den `M>=4` olaylari indirir.
- Cikti: `data/raw/earthquakes_m4_2000_2026.parquet`

2. `generate_planet_features.py`
- Deprem dakikalarinda gezegen vektorlerini SPICE ile uretir.
- Ayni sayida negatif ornek dakika (deprem olmayan) uretir.
- Cikti: `data/processed/planet_features_eq_and_controls.parquet`

3. `build_model_dataset.py`
- Ozellikler ile deprem etiketini birlestirir.
- Cikti: `data/processed/model_dataset.parquet`

4. `train_baseline.py`
- Baslangic modeli: `HistGradientBoostingClassifier`
- Metrikler: ROC-AUC, PR-AUC, permutation p-value
- Cikti: `models/baseline_metrics.json`

5. `train_time_split.py`
- Daha gercekci test: zamani gecmise gore ayirir (cutoff: `2018-01-01`)
- Cikti: `models/time_split_metrics.json`

6. `train_location_models.py`
- Lokasyon girdili prototip modelleri egitir:
- Siniflandirma: `P(M>=4 | dakika, lokasyon)`
- Regresyon: Olay olursa tahmini buyukluk
- Cikti:
  - `models/location_classifier.joblib`
  - `models/magnitude_regressor.joblib`
  - `models/location_models_meta.json`

7. `predict_location_cli.py`
- Girdi: `--lat --lon --start --end --top-k`
- Cikti: en yuksek riskli dakikalar
- Varsayilan dosya: `models/location_predictions_topk.csv`

8. `train_time_rolling_cv.py`
- Rolling zaman dogrulama (genisleyen train, ileri tarih test)
- Cikti: `models/rolling_cv_metrics.json`

9. `seismic_features.py`
- Lokasyon icin gecmis sismisite feature'lari:
  - son 1/7/30/365 gunde 300 km icinde olay sayisi
  - son 365 gunde max/ortalama magnitud
  - en son olaya gecen gun

10. `mine/location_predict_demo.ipynb`
- Tek notebook akisinda lokasyon tahmin denemesi

11. Kalibrasyon ve belirsizlik
- `train_location_models.py` artik su ciktilari da uretir:
  - `models/location_classifier_calibrator.joblib`
  - `models/location_classifier_ensemble.joblib`
  - `models/magnitude_regressor_q10.joblib`
  - `models/magnitude_regressor_q90.joblib`
- Siniflandirma: kalibre olasilik + ensemble tabanli 95% bant
- Magnitud: q10-q90 araligi (yaklasik belirsizlik)

12. Grid risk haritasi
- `predict_grid_risk.py` tek bir UTC zamaninda lat/lon grid uzerinde risk ve magnitud tahmini uretir
- Cikti:
  - `models/grid_risk_map.csv`
  - `models/grid_risk_map_topk.csv`
- Tek komut pipeline:
  - `run_grid_pipeline.py`
  - Ornek:
    - `python run_grid_pipeline.py --time 2026-04-01T00:00:00Z --lat-min 30 --lat-max 50 --lon-min 20 --lon-max 45 --step 5 --top-k 20 --out-csv ../models/grid_risk_map_onecmd.csv --out-png ../models/grid_risk_map_onecmd.png`

13. Sade notebook arayuzu
- `mine/one_click_prediction.ipynb`
- Tek hucrede girilen `lat/lon/start/end` ile tahmin ciktisi alir

14. Grid isi haritasi render
- `pipeline/render_grid_heatmap.py`
- Grid CSV'den PNG isi haritasi uretir
- Ornek:
  - `python render_grid_heatmap.py --in ../models/grid_risk_map_sample.csv --out ../models/grid_risk_map_sample.png`

15. Interaktif notebook form
- `mine/interactive_prediction.ipynb`
- `ipywidgets` varsa form (lat/lon/start/end/top-k) ile calisir
- `ipywidgets` yoksa parametre hucreden manuel girilerek fallback yapar

## 4) Calistirma Sirasi

```bash
cd /Users/engin/D/mygithub/6.MyProjects/Planets/pipeline
python fetch_earthquakes.py
python generate_planet_features.py
python build_model_dataset.py
python train_baseline.py
python train_time_split.py
python train_location_models.py
python train_time_rolling_cv.py
python predict_location_cli.py --lat 39.93 --lon 32.85 --start 2026-04-01T00:00:00Z --end 2026-04-03T00:00:00Z --top-k 20
python predict_grid_risk.py --time 2026-04-01T00:00:00Z --lat-min 30 --lat-max 50 --lon-min 20 --lon-max 45 --step 5 --top-k 100
```

## 5) Sonucu Nasil Yorumlamalisin

Anlamli sayilabilecek minimum kosul:
- ROC-AUC belirgin sekilde 0.5 ustu (ornegin >= 0.6)
- PR-AUC, pozitif oranina gore anlamli artis gostermeli
- permutation p-value < 0.05

Bu kosullar saglanmiyorsa, model rastgele seviyeye yakindir.

## 6) Kullanicidan Lokasyon Alip Tahmin Etme

Mevcut bilimsel risk: Sadece gezegen konumlariyla "hangi tarihte hangi buyukluk" tahmini yapmak gercekci degil.

Eger yine de prototip istenirse:
1. Girdi: `latitude`, `longitude`, `tarih araligi`
2. Her dakika icin gezegen ozellikleri + bolgeye ait tarihsel sismik ozellikler
3. Cikti A: `P(M>=4 | dakika, lokasyon)` (siniflandirma)
4. Cikti B: Olasilik yuksek dakikalarda buyukluk icin ayri regresyon modeli

## 7) Guvenlik Notu

`planet.ipynb` dosyasinda acik API anahtari/secret bulunuyor. Bu anahtarlarin iptal edilip yeniden uretilmesi gerekir.
