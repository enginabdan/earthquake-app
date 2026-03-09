from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium
try:
    from streamlit_geolocation import streamlit_geolocation
except Exception:
    streamlit_geolocation = None

ROOT = Path(__file__).resolve().parent
PIPE = ROOT / "pipeline"
if str(PIPE) not in sys.path:
    sys.path.insert(0, str(PIPE))

from pipeline.config import DATA_RAW, MODELS
from pipeline.coord_dms import format_lat_dms, format_lon_dms, parse_coordinate
from pipeline.generate_planet_features import build_feature_row, load_kernels
from pipeline.seismic_features import compute_history_features
from pipeline.train_location_models import location_features


st.set_page_config(page_title="Earthquake Risk Demo", layout="wide")

LANG_OPTIONS = [
    "Almanca",
    "Fransızca",
    "İngilizce",
    "İspanyolca",
    "Rusça",
    "Türkçe",
]

LANG_CODE = {
    "Almanca": "de",
    "Fransızca": "fr",
    "İngilizce": "en",
    "İspanyolca": "es",
    "Rusça": "ru",
    "Türkçe": "tr",
}

TXT = {
    "title": {
        "en": "Earthquake Risk Demo",
        "de": "Erdbebenrisiko-Demo",
        "fr": "Demo de risque sismique",
        "es": "Demo de riesgo sísmico",
        "ru": "Демо риска землетрясений",
        "tr": "Deprem Risk Demosu",
    },
    "caption": {
        "en": "This interface is a research prototype and does not provide exact earthquake predictions.",
        "de": "Diese Oberfläche ist ein Forschungsprototyp und liefert keine exakten Erdbebenvorhersagen.",
        "fr": "Cette interface est un prototype de recherche et ne fournit pas de prédictions exactes des tremblements de terre.",
        "es": "Esta interfaz es un prototipo de investigación y no proporciona predicciones exactas de terremotos.",
        "ru": "Этот интерфейс является исследовательским прототипом и не дает точных прогнозов землетрясений.",
        "tr": "Bu arayüz bir araştırma prototipidir ve kesin deprem tahmini vermez.",
    },
    "language": {
        "en": "Language",
        "de": "Sprache",
        "fr": "Langue",
        "es": "Idioma",
        "ru": "Язык",
        "tr": "Dil",
    },
    "tab_location": {"en": "Location Forecast", "de": "Standortprognose", "fr": "Prévision du lieu", "es": "Pronóstico por ubicación", "ru": "Прогноз по местоположению", "tr": "Lokasyon Tahmini"},
    "tab_grid": {"en": "Grid Heatmap", "de": "Raster-Wärmekarte", "fr": "Carte thermique en grille", "es": "Mapa de calor en cuadrícula", "ru": "Тепловая карта сетки", "tr": "Grid Isı Haritası"},
    "location_help": {
        "en": "Allow location access to center the map on your current position. You can click the map to refine the point.",
        "de": "Erlaube den Standortzugriff, um die Karte auf deine aktuelle Position zu zentrieren. Du kannst auf die Karte klicken, um den Punkt zu verfeinern.",
        "fr": "Autorisez l’accès à la localisation pour centrer la carte sur votre position actuelle. Vous pouvez cliquer sur la carte pour affiner le point.",
        "es": "Permite el acceso a la ubicación para centrar el mapa en tu posición actual. Puedes hacer clic en el mapa para ajustar el punto.",
        "ru": "Разрешите доступ к геолокации, чтобы центрировать карту на вашем текущем местоположении. Можно кликнуть по карте для уточнения точки.",
        "tr": "Haritayı mevcut konumuna ortalamak için konum izni ver. Noktayı hassaslaştırmak için haritaya tıklayabilirsin.",
    },
    "install_geo": {
        "en": "Install 'streamlit-geolocation' to auto-center the map on your current location.",
        "de": "Installiere 'streamlit-geolocation', um die Karte automatisch auf deinen aktuellen Standort zu zentrieren.",
        "fr": "Installez 'streamlit-geolocation' pour centrer automatiquement la carte sur votre position actuelle.",
        "es": "Instala 'streamlit-geolocation' para centrar automáticamente el mapa en tu ubicación actual.",
        "ru": "Установите 'streamlit-geolocation', чтобы автоматически центрировать карту на вашем текущем местоположении.",
        "tr": "Haritayı mevcut konumuna otomatik ortalamak için 'streamlit-geolocation' kur.",
    },
    "selected_location": {"en": "Selected Location", "de": "Ausgewählter Standort", "fr": "Lieu sélectionné", "es": "Ubicación seleccionada", "ru": "Выбранное местоположение", "tr": "Seçili Konum"},
    "lat_decimal": {"en": "Latitude (decimal)", "de": "Breitengrad (dezimal)", "fr": "Latitude (décimal)", "es": "Latitud (decimal)", "ru": "Широта (десятичная)", "tr": "Enlem (ondalık)"},
    "lon_decimal": {"en": "Longitude (decimal)", "de": "Längengrad (dezimal)", "fr": "Longitude (décimal)", "es": "Longitud (decimal)", "ru": "Долгота (десятичная)", "tr": "Boylam (ondalık)"},
    "topk": {"en": "Top-K", "de": "Top-K", "fr": "Top-K", "es": "Top-K", "ru": "Top-K", "tr": "Top-K"},
    "min_gap": {"en": "Minimum gap between shown rows (minutes)", "de": "Mindestabstand zwischen angezeigten Zeilen (Minuten)", "fr": "Intervalle minimum entre les lignes affichées (minutes)", "es": "Separación mínima entre filas mostradas (minutos)", "ru": "Минимальный интервал между показанными строками (минуты)", "tr": "Gösterilen satırlar arası minimum fark (dakika)"},
    "start_date": {"en": "Start date (UTC)", "de": "Startdatum (UTC)", "fr": "Date de début (UTC)", "es": "Fecha de inicio (UTC)", "ru": "Дата начала (UTC)", "tr": "Başlangıç tarihi (UTC)"},
    "end_date": {"en": "End date (UTC)", "de": "Enddatum (UTC)", "fr": "Date de fin (UTC)", "es": "Fecha de fin (UTC)", "ru": "Дата окончания (UTC)", "tr": "Bitiş tarihi (UTC)"},
    "range_warn": {"en": "The selected range is {days} days. Computation may take longer.", "de": "Der gewählte Bereich beträgt {days} Tage. Die Berechnung kann länger dauern.", "fr": "La plage sélectionnée est de {days} jours. Le calcul peut prendre plus de temps.", "es": "El rango seleccionado es de {days} días. El cálculo puede tardar más.", "ru": "Выбранный диапазон: {days} дней. Расчет может занять больше времени.", "tr": "Seçilen aralık {days} gün. Hesaplama daha uzun sürebilir."},
    "run_forecast": {"en": "Run Forecast", "de": "Prognose ausführen", "fr": "Lancer la prévision", "es": "Ejecutar pronóstico", "ru": "Запустить прогноз", "tr": "Tahmini Çalıştır"},
    "spinner_forecast": {"en": "Computing forecast, please wait...", "de": "Prognose wird berechnet, bitte warten...", "fr": "Calcul de la prévision, veuillez patienter...", "es": "Calculando pronóstico, por favor espera...", "ru": "Выполняется расчет прогноза, пожалуйста, подождите...", "tr": "Tahmin hesaplanıyor, lütfen bekleyin..."},
    "no_rows": {"en": "No rows left after applying the minimum-gap filter. Reduce the gap and try again.", "de": "Nach dem Mindestabstandsfilter sind keine Zeilen übrig. Verringere den Abstand und versuche es erneut.", "fr": "Aucune ligne restante après application du filtre d’intervalle minimum. Réduisez l’intervalle et réessayez.", "es": "No quedan filas después de aplicar el filtro de separación mínima. Reduce la separación e inténtalo de nuevo.", "ru": "После применения фильтра минимального интервала строк не осталось. Уменьшите интервал и попробуйте снова.", "tr": "Minimum fark filtresi sonrası satır kalmadı. Farkı azaltıp tekrar dene."},
    "forecast_done": {"en": "Forecast completed.", "de": "Prognose abgeschlossen.", "fr": "Prévision terminée.", "es": "Pronóstico completado.", "ru": "Прогноз завершен.", "tr": "Tahmin tamamlandı."},
    "forecast_disclaimer": {"en": "This output is probabilistic research output, not a deterministic earthquake prediction. Use it only as an exploratory signal.", "de": "Dieses Ergebnis ist ein probabilistisches Forschungsergebnis und keine deterministische Erdbebenvorhersage. Verwende es nur als exploratives Signal.", "fr": "Ce résultat est une sortie probabiliste de recherche, pas une prédiction déterministe de séisme. Utilisez-le uniquement comme signal exploratoire.", "es": "Este resultado es una salida probabilística de investigación, no una predicción determinista de terremotos. Úsalo solo como señal exploratoria.", "ru": "Этот результат является вероятностным исследовательским выводом, а не детерминированным прогнозом землетрясения. Используйте его только как исследовательский сигнал.", "tr": "Bu çıktı olasılıksal bir araştırma çıktısıdır, deterministik deprem tahmini değildir. Sadece keşif amaçlı sinyal olarak kullan."},
    "error": {"en": "Error", "de": "Fehler", "fr": "Erreur", "es": "Error", "ru": "Ошибка", "tr": "Hata"},
    "risk_low": {"en": "Low", "de": "Niedrig", "fr": "Faible", "es": "Bajo", "ru": "Низкий", "tr": "Düşük"},
    "risk_medium": {"en": "Medium", "de": "Mittel", "fr": "Moyen", "es": "Medio", "ru": "Средний", "tr": "Orta"},
    "risk_high": {"en": "High", "de": "Hoch", "fr": "Élevé", "es": "Alto", "ru": "Высокий", "tr": "Yüksek"},
    "grid_timestamp": {"en": "Grid timestamp (UTC)", "de": "Raster-Zeitstempel (UTC)", "fr": "Horodatage de la grille (UTC)", "es": "Marca temporal de la cuadrícula (UTC)", "ru": "Временная метка сетки (UTC)", "tr": "Grid zaman damgası (UTC)"},
    "step_deg": {"en": "Step (degrees)", "de": "Schritt (Grad)", "fr": "Pas (degrés)", "es": "Paso (grados)", "ru": "Шаг (градусы)", "tr": "Adım (derece)"},
    "lat_min": {"en": "Lat Min (DMS)", "de": "Min. Breite (DMS)", "fr": "Lat min (DMS)", "es": "Lat mín (DMS)", "ru": "Мин. широта (DMS)", "tr": "Enlem Min (DMS)"},
    "lat_max": {"en": "Lat Max (DMS)", "de": "Max. Breite (DMS)", "fr": "Lat max (DMS)", "es": "Lat máx (DMS)", "ru": "Макс. широта (DMS)", "tr": "Enlem Max (DMS)"},
    "lon_min": {"en": "Lon Min (DMS)", "de": "Min. Länge (DMS)", "fr": "Lon min (DMS)", "es": "Lon mín (DMS)", "ru": "Мин. долгота (DMS)", "tr": "Boylam Min (DMS)"},
    "lon_max": {"en": "Lon Max (DMS)", "de": "Max. Länge (DMS)", "fr": "Lon max (DMS)", "es": "Lon máx (DMS)", "ru": "Макс. долгота (DMS)", "tr": "Boylam Max (DMS)"},
    "generate_grid": {"en": "Generate Grid", "de": "Raster erzeugen", "fr": "Générer la grille", "es": "Generar cuadrícula", "ru": "Построить сетку", "tr": "Grid Üret"},
    "spinner_grid": {"en": "Computing grid, please wait...", "de": "Raster wird berechnet, bitte warten...", "fr": "Calcul de la grille, veuillez patienter...", "es": "Calculando cuadrícula, por favor espera...", "ru": "Выполняется расчет сетки, пожалуйста, подождите...", "tr": "Grid hesaplanıyor, lütfen bekleyin..."},
    "xlabel_lon": {"en": "Longitude", "de": "Längengrad", "fr": "Longitude", "es": "Longitud", "ru": "Долгота", "tr": "Boylam"},
    "ylabel_lat": {"en": "Latitude", "de": "Breitengrad", "fr": "Latitude", "es": "Latitud", "ru": "Широта", "tr": "Enlem"},
    "grid_title": {"en": "Grid Risk Heatmap", "de": "Raster-Risikowärmekarte", "fr": "Carte thermique du risque en grille", "es": "Mapa de calor de riesgo en cuadrícula", "ru": "Тепловая карта риска по сетке", "tr": "Grid Risk Isı Haritası"},
    "cbar_label": {"en": "Calibrated P(M>=4)", "de": "Kalibriertes P(M>=4)", "fr": "P(M>=4) calibrée", "es": "P(M>=4) calibrada", "ru": "Калиброванная P(M>=4)", "tr": "Kalibre P(M>=4)"},
}


def tr(key: str, lang: str, **kwargs) -> str:
    s = TXT.get(key, {}).get(lang, TXT.get(key, {}).get("en", key))
    return s.format(**kwargs) if kwargs else s


_hdr_l, _hdr_r = st.columns([0.78, 0.22])
with _hdr_r:
    lang_label = st.selectbox(
        "Language",
        options=LANG_OPTIONS,
        index=LANG_OPTIONS.index("İngilizce"),
        key="ui_lang_option",
    )
LANG = LANG_CODE[lang_label]

st.title(tr("title", LANG))
st.caption(tr("caption", LANG))


@st.cache_resource
def load_models():
    clf = joblib.load(MODELS / "location_classifier.joblib")
    calibrator = joblib.load(MODELS / "location_classifier_calibrator.joblib")
    ensemble = joblib.load(MODELS / "location_classifier_ensemble.joblib")
    reg = joblib.load(MODELS / "magnitude_regressor.joblib")
    reg_q10 = joblib.load(MODELS / "magnitude_regressor_q10.joblib")
    reg_q90 = joblib.load(MODELS / "magnitude_regressor_q90.joblib")
    feature_cols = json.loads((MODELS / "location_models_meta.json").read_text(encoding="utf-8"))["feature_columns"]
    return clf, calibrator, ensemble, reg, reg_q10, reg_q90, feature_cols


@st.cache_data
def load_eq_catalog():
    eq = pd.read_parquet(DATA_RAW / "earthquakes_m4_2000_2026.parquet")
    eq["time_utc"] = pd.to_datetime(eq["time_utc"], utc=True).dt.floor("min")
    return eq


def build_location_predictions(lat: float, lon: float, start_utc: pd.Timestamp, end_utc: pd.Timestamp, top_k: int):
    clf, calibrator, ensemble, reg, reg_q10, reg_q90, feature_cols = load_models()
    eq = load_eq_catalog()

    minutes = pd.date_range(start=start_utc, end=end_utc, freq="min", tz="UTC")
    if len(minutes) == 0:
        return pd.DataFrame()

    rows = []
    loc = location_features(lat, lon)

    load_kernels()
    try:
        for ts in minutes:
            feat = build_feature_row(ts)
            feat.update(loc)
            rows.append(feat)
    finally:
        import spiceypy as spice

        spice.kclear()

    df = pd.DataFrame(rows)
    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True)
    hist_input = pd.DataFrame({"time_utc": df["time_utc"], "latitude": lat, "longitude": lon})
    hist = compute_history_features(hist_input, eq, radius_km=300.0)
    df = pd.concat([df.reset_index(drop=True), hist.reset_index(drop=True)], axis=1)

    X = df[feature_cols]
    p_cal = calibrator.predict_proba(X)[:, 1]
    ens_probs = np.vstack([m.predict_proba(X)[:, 1] for m in ensemble])
    p_std = ens_probs.std(axis=0)

    df["p_eq_m4_plus"] = p_cal
    df["p_eq_m4_plus_low95"] = np.clip(p_cal - 1.96 * p_std, 0.0, 1.0)
    df["p_eq_m4_plus_high95"] = np.clip(p_cal + 1.96 * p_std, 0.0, 1.0)
    df["pred_magnitude_if_event"] = reg.predict(X)
    df["pred_magnitude_q10"] = reg_q10.predict(X)
    df["pred_magnitude_q90"] = reg_q90.predict(X)
    df["latitude_dms"] = format_lat_dms(lat)
    df["longitude_dms"] = format_lon_dms(lon)

    return df.sort_values("p_eq_m4_plus", ascending=False)


def pick_top_with_min_gap(df: pd.DataFrame, top_k: int, min_gap_minutes: int) -> pd.DataFrame:
    if df.empty:
        return df
    if min_gap_minutes <= 0:
        return df.head(top_k)

    selected = []
    min_gap = pd.Timedelta(minutes=int(min_gap_minutes))
    for row in df.itertuples(index=False):
        t = pd.Timestamp(row.time_utc)
        if all(abs(t - pd.Timestamp(r["time_utc"])) >= min_gap for r in selected):
            selected.append(dict(zip(df.columns, row)))
        if len(selected) >= top_k:
            break
    return pd.DataFrame(selected)


def build_grid_predictions(
    ts_utc: pd.Timestamp,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    step: float,
):
    clf, calibrator, ensemble, reg, reg_q10, reg_q90, feature_cols = load_models()
    eq = load_eq_catalog()

    lats = np.arange(lat_min, lat_max + 1e-9, step)
    lons = np.arange(lon_min, lon_max + 1e-9, step)

    load_kernels()
    try:
        base = build_feature_row(ts_utc)
    finally:
        import spiceypy as spice

        spice.kclear()

    rows = []
    for lat in lats:
        for lon in lons:
            r = dict(base)
            r.update(location_features(float(lat), float(lon)))
            rows.append(r)

    df = pd.DataFrame(rows)
    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True)
    hist_input = df[["time_utc", "latitude", "longitude"]].copy()
    hist = compute_history_features(hist_input, eq, radius_km=300.0)
    df = pd.concat([df.reset_index(drop=True), hist.reset_index(drop=True)], axis=1)

    X = df[feature_cols]
    p_cal = calibrator.predict_proba(X)[:, 1]
    ens_probs = np.vstack([m.predict_proba(X)[:, 1] for m in ensemble])
    p_std = ens_probs.std(axis=0)

    df["p_eq_m4_plus"] = p_cal
    df["p_eq_m4_plus_low95"] = np.clip(p_cal - 1.96 * p_std, 0.0, 1.0)
    df["p_eq_m4_plus_high95"] = np.clip(p_cal + 1.96 * p_std, 0.0, 1.0)
    df["pred_magnitude_if_event"] = reg.predict(X)
    df["pred_magnitude_q10"] = reg_q10.predict(X)
    df["pred_magnitude_q90"] = reg_q90.predict(X)
    df["latitude_dms"] = df["latitude"].apply(format_lat_dms)
    df["longitude_dms"] = df["longitude"].apply(format_lon_dms)
    return df


tab1, tab2 = st.tabs([tr("tab_location", LANG), tr("tab_grid", LANG)])

with tab1:
    st.subheader(tr("tab_location", LANG))
    if "selected_lat" not in st.session_state:
        st.session_state["selected_lat"] = 39.8283
    if "selected_lon" not in st.session_state:
        st.session_state["selected_lon"] = -98.5795

    st.markdown(tr("location_help", LANG))
    if streamlit_geolocation is not None:
        browser_loc = streamlit_geolocation()
        if browser_loc and browser_loc.get("latitude") is not None and browser_loc.get("longitude") is not None:
            st.session_state["selected_lat"] = float(browser_loc["latitude"])
            st.session_state["selected_lon"] = float(browser_loc["longitude"])
    else:
        st.info(tr("install_geo", LANG))

    m = folium.Map(
        location=[st.session_state["selected_lat"], st.session_state["selected_lon"]],
        zoom_start=5,
        tiles="OpenStreetMap",
        control_scale=True,
    )
    folium.Marker(
        [st.session_state["selected_lat"], st.session_state["selected_lon"]],
        tooltip=tr("selected_location", LANG),
    ).add_to(m)
    folium.LatLngPopup().add_to(m)
    map_data = st_folium(m, height=420, width=None, key="location_picker")
    if map_data and map_data.get("last_clicked"):
        st.session_state["selected_lat"] = float(map_data["last_clicked"]["lat"])
        st.session_state["selected_lon"] = float(map_data["last_clicked"]["lng"])

    lat = float(st.session_state["selected_lat"])
    lon = float(st.session_state["selected_lon"])
    c1, c2 = st.columns(2)
    c1.metric(tr("lat_decimal", LANG), f"{lat:.6f}")
    c2.metric(tr("lon_decimal", LANG), f"{lon:.6f}")

    c5, c6 = st.columns(2)
    top_k = c5.number_input(tr("topk", LANG), min_value=5, max_value=200, value=20, step=5)
    min_gap_min = c6.number_input(tr("min_gap", LANG), min_value=0, max_value=1440, value=360, step=30)

    d1, d2 = st.columns(2)
    start_date = d1.date_input(tr("start_date", LANG), datetime(2026, 4, 1).date())
    end_date = d2.date_input(tr("end_date", LANG), datetime(2026, 4, 3).date())
    day_span = (end_date - start_date).days + 1
    if day_span > 31:
        st.warning(tr("range_warn", LANG, days=day_span))

    if st.button(tr("run_forecast", LANG), type="primary"):
        try:
            start_ts = pd.Timestamp(datetime.combine(start_date, datetime.min.time(), tzinfo=timezone.utc))
            end_ts = pd.Timestamp(datetime.combine(end_date, datetime.min.time(), tzinfo=timezone.utc) + timedelta(hours=23, minutes=59))
            with st.spinner(tr("spinner_forecast", LANG)):
                ranked = build_location_predictions(lat, lon, start_ts, end_ts, int(top_k))
                out = pick_top_with_min_gap(ranked, int(top_k), int(min_gap_min))
                if out.empty:
                    st.warning(tr("no_rows", LANG))
                    st.stop()

                out["uncertainty_width"] = out["p_eq_m4_plus_high95"] - out["p_eq_m4_plus_low95"]
                out["risk_level"] = pd.cut(
                    out["p_eq_m4_plus"],
                    bins=[-0.001, 0.33, 0.66, 1.001],
                    labels=[tr("risk_low", LANG), tr("risk_medium", LANG), tr("risk_high", LANG)],
                )
                out["confidence_level"] = pd.cut(
                    out["uncertainty_width"],
                    bins=[-0.001, 0.20, 0.40, 10.0],
                    labels=[tr("risk_high", LANG), tr("risk_medium", LANG), tr("risk_low", LANG)],
                )

            st.success(tr("forecast_done", LANG))
            st.warning(tr("forecast_disclaimer", LANG))
            st.dataframe(
                out[
                    [
                        "time_utc",
                        "latitude",
                        "longitude",
                        "p_eq_m4_plus",
                        "p_eq_m4_plus_low95",
                        "p_eq_m4_plus_high95",
                        "risk_level",
                        "confidence_level",
                        "pred_magnitude_if_event",
                        "pred_magnitude_q10",
                        "pred_magnitude_q90",
                    ]
                ],
                use_container_width=True,
            )
        except Exception as exc:
            st.error(f"{tr('error', LANG)}: {exc}")


with tab2:
    st.subheader(tr("tab_grid", LANG))
    g1, g2, g3 = st.columns(3)
    grid_time = g1.text_input(tr("grid_timestamp", LANG), "2026-04-01T00:00:00Z")
    step = g2.number_input(tr("step_deg", LANG), min_value=0.5, max_value=10.0, value=5.0, step=0.5)
    grid_topk = g3.number_input(tr("topk", LANG), min_value=5, max_value=500, value=20, step=5)

    b1, b2, b3, b4 = st.columns(4)
    lat_min_in = b1.text_input(tr("lat_min", LANG), "30.00.00N")
    lat_max_in = b2.text_input(tr("lat_max", LANG), "50.00.00N")
    lon_min_in = b3.text_input(tr("lon_min", LANG), "20.00.00E")
    lon_max_in = b4.text_input(tr("lon_max", LANG), "45.00.00E")

    if st.button(tr("generate_grid", LANG)):
        try:
            ts = pd.Timestamp(grid_time)
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            else:
                ts = ts.tz_convert("UTC")

            lat_min = parse_coordinate(lat_min_in, is_lat=True)
            lat_max = parse_coordinate(lat_max_in, is_lat=True)
            lon_min = parse_coordinate(lon_min_in, is_lat=False)
            lon_max = parse_coordinate(lon_max_in, is_lat=False)

            with st.spinner(tr("spinner_grid", LANG)):
                grid = build_grid_predictions(ts, lat_min, lat_max, lon_min, lon_max, float(step))
                top = grid.sort_values("p_eq_m4_plus", ascending=False).head(int(grid_topk))

            fig, ax = plt.subplots(figsize=(10, 5))
            sc = ax.scatter(grid["longitude"], grid["latitude"], c=grid["p_eq_m4_plus"], cmap="inferno", s=55, alpha=0.9)
            ax.set_xlabel(tr("xlabel_lon", LANG))
            ax.set_ylabel(tr("ylabel_lat", LANG))
            ax.set_title(tr("grid_title", LANG))
            plt.colorbar(sc, ax=ax, label=tr("cbar_label", LANG))
            st.pyplot(fig, clear_figure=True, use_container_width=True)

            st.dataframe(
                top[
                    [
                        "latitude_dms",
                        "longitude_dms",
                        "p_eq_m4_plus",
                        "p_eq_m4_plus_low95",
                        "p_eq_m4_plus_high95",
                        "pred_magnitude_if_event",
                    ]
                ],
                use_container_width=True,
            )
        except Exception as exc:
            st.error(f"{tr('error', LANG)}: {exc}")
