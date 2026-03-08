# Earthquake Mini Project

Bu klasor, deprem + gezegen konumu analizi icin tek basina kullanilabilir mini proje yapisidir.

Koordinat formati tum projede DMS olarak kullanilabilir:
- Ornek enlem: `39.55.48N`
- Ornek boylam: `32.51.00E`
- Decimal da kabul edilir, fakat ciktilarda DMS kolonlari da uretilir (`latitude_dms`, `longitude_dms`).

## Klasor Yapisi

- `pipeline/`: veri cekme, ozellik uretme, model egitimi ve tahmin scriptleri
- `data/`: ham ve islenmis veri
- `models/`: egitilmis modeller ve metrikler
- `mine/`: demo notebook'lar
- `PROJECT_REBUILD_TR.md`: teknik yol haritasi ve notlar

## Kurulum

`requirements.txt` dosyasi bu klasordedir.

```bash
cd /Users/engin/D/mygithub/earthquake
python3 -m pip install -r requirements.txt
```

Alternatif:

```bash
cd /Users/engin/D/mygithub/earthquake
make install
```

## Uctan Uca Calistirma

```bash
cd /Users/engin/D/mygithub/earthquake
make train
```

## Lokasyon Tahmini

```bash
cd /Users/engin/D/mygithub/earthquake
make predict
```

## Grid Risk + Isi Haritasi (Tek Komut)

```bash
cd /Users/engin/D/mygithub/earthquake
make grid
```

## Makefile Kisayollari

```bash
cd /Users/engin/D/mygithub/earthquake
make help
```

Ornek parametre override:

```bash
make predict LAT=41.00.00N LON=29.00.00E START=2026-05-01T00:00:00Z END=2026-05-02T00:00:00Z TOPK=50
make grid GRID_TIME=2026-05-01T00:00:00Z LAT_MIN=35.00.00N LAT_MAX=45.00.00N LON_MIN=25.00.00E LON_MAX=40.00.00E STEP=2 GRID_TOPK=100
```

## Son Kullanici Arayuzu (Web)

En kolay kullanim:

```bash
cd /Users/engin/D/mygithub/earthquake
make app
```

Acilan arayuzde:
- `Lokasyon Tahmini` sekmesi: Browser location izin verilirse harita kullanicinin anlik konumunda acilir; haritadan tiklayarak lokasyon secip tarih araligi ile tablo sonucu alirsin
- `Grid Isi Haritasi` sekmesi: DMS sinirlari girerek harita ve top-k tablo alirsin

## 7/24 Canli Yayin (Ayni UI ile)

Bu proje icin asagidaki deployment dosyalari eklendi:
- `Dockerfile`
- `docker-compose.yml`
- `deploy/nginx/nginx.conf`
- `deploy/systemd/earthquake-streamlit.service`
- `deploy/scripts/deploy.sh`

1) Sunucuda proje klasorunu ac:

```bash
cd /opt/earthquake
```

2) SSL sertifika yollarini ve domain'i vererek stack'i kaldir:

```bash
DOMAIN=app.yourdomain.com \
CERT_FULLCHAIN=/etc/letsencrypt/live/app.yourdomain.com/fullchain.pem \
CERT_PRIVKEY=/etc/letsencrypt/live/app.yourdomain.com/privkey.pem \
./deploy/scripts/deploy.sh
```

3) Mevcut web sitende iframe ile goster:

```html
<iframe
  src="https://app.yourdomain.com"
  style="width:100%;height:100vh;border:0;"
  loading="lazy"
></iframe>
```

Opsiyonel systemd:

```bash
sudo cp deploy/systemd/earthquake-streamlit.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable earthquake-streamlit
sudo systemctl start earthquake-streamlit
```

## Notebook'lar

- `mine/interactive_prediction.ipynb`: form tabanli tahmin ve grid heatmap
- `mine/one_click_prediction.ipynb`: tek hucrede tahmin
- `mine/location_predict_demo.ipynb`: CLI tabanli demo
