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

### Render Uzerinden Yayin

Bu repo Render Blueprint ile yayinlanabilir (`render.yaml`):

1) Render'da **New + -> Blueprint** sec
2) Bu GitHub reposunu bagla
3) `earthquake-app` web service olustugunda deploy et
4) Canli URL: `https://<render-service>.onrender.com`

### GoDaddy DNS (Custom Domain)

Render dashboard -> Settings -> Custom Domains:

1) Domain ekle (ornek: `app.senin-domain.com`)
2) Render'in verdigi hedefe GoDaddy'de `CNAME` ac:
   - Host: `app`
   - Type: `CNAME`
   - Points to: Render'in verdigi hedef (`xxxxx.onrender.com`)
3) SSL Render tarafinda otomatik aktif olur (DNS propagate sonrasi)

### Google Sites Icine Gommek

Google Sites sayfasinda Embed -> URL:

```html
https://app.senin-domain.com
```

## Notebook'lar

Notebook'lar temizlendi. Uygulama ve tum akislar `make` komutlari ve `app.py` uzerinden calisir.
