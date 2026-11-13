# Dokumentasi Dataset Penjalaran Kebakaran Hutan dan Lahan Kalimantan (2017–2026)

Dokumentasi ini merupakan pembaruan teknis dari `DATASET_CARD.md` yang dirapikan dan disesuaikan dengan metodologi partisi data terbaru (**`hybrid_tile_disjoint`**).

---

## 1. Ringkasan & Cakupan Dataset

**Kalimantan Wildfire Spread Dataset** adalah dataset *spatiotemporal* siap-latih untuk pemodelan prediksi penjalaran kebakaran hutan dan lahan berbasis *deep learning* (misalnya U-Net, SegFormer, Swin-UNet) maupun *Physics-Guided Neural Cellular Automata* (`Neural-CA`).

| Atribut | Spesifikasi |
| :--- | :--- |
| **Wilayah Cakupan** | Seluruh Kalimantan, Indonesia ($4{,}5^\circ\text{LS} - 4{,}5^\circ\text{LU}$, $108{,}5^\circ\text{BT} - 119{,}5^\circ\text{BT}$): Kalimantan Barat, Kalimantan Tengah, Kalimantan Selatan, Kalimantan Timur, Kalimantan Utara |
| **Periode Temporal** | 10 Juli 2017 – 31 Maret 2026 (resolusi harian) |
| **Total Sampel** | **5.283** *patch* kejadian-hari ($64 \times 64$ piksel) |
| **Resolusi Spasial** | $375\text{ m} \times 375\text{ m}$ per piksel ($1\text{ piksel} = 14{,}0625\text{ ha}$; $1\text{ patch} = 24\text{ km} \times 24\text{ km}$) |
| **Sistem Koordinat (CRS)** | Lambert Azimuthal Equal Area (`+proj=laea +lat_0=0 +lon_0=114 +datum=WGS84 +units=m +no_defs`) |
| **Mode ROI** | `active_cluster` — klaster api aktif yang bersebelahan pada hari $t$ ($\ge 5$ piksel) dipusatkan pada koordinat lokal $(32, 32)$ |
| **Horizon Prediksi** | Multi-horizon: $+24\text{ jam}$ ($t+1$), $+48\text{ jam}$ ($t+2$), dan $+72\text{ jam}$ ($t+3$) |
| **Protokol Partisi Default** | `hybrid_tile_disjoint` (`train`: 3.699 | `val`: 790 | `test`: 794) |

---

## 2. Karakteristik Masalah & Tantangan Pemodelan

Kebakaran di Kalimantan didominasi oleh kebakaran lahan gambut tropis dan vegetasi terbuka yang memburuk saat fase kering El Niño Southern Oscillation (ENSO) dan Indian Ocean Dipole (IOD) positif (seperti tahun 2019 dan 2023). Dataset ini dirancang untuk merepresentasikan dan mengatasi empat tantangan utama pemodelan:

1. **Ketidakseimbangan Kelas Ekstrem (*Extreme Class Imbalance*)**:  
   Pada gabungan piksel valid `train` + `val`, piksel baru terbakar (`y_new`) hanya mencakup **0,409%** (rasio $\approx 244:1$, yaitu $0{,}233\%$ di $t+1$, $0{,}426\%$ di $t+2$, dan $0{,}579\%$ di $t+3$), sedangkan target kumulatif (`y`) mencakup **3,288%** (rasio $\approx 29:1$). Pelatihan disarankan menggunakan fungsi *loss* yang tahan ketidakseimbangan (*masked* Focal/Dice/Tversky loss).
2. ***Zero-Inflation* pada Level *Patch***:  
   Meskipun terdapat api aktif pada hari $t$, sekitar **34,4%** *patch* pada $t+1$, **26,8%** pada $t+2$, dan **24,0%** pada $t+3$ tidak menunjukkan perluasan piksel baru (`y_new = 0`).
3. **Oklusi Awan & Masking Spasial (`valid`)**:  
   Tutupan awan tropis, badan air permanen, area di luar batas AOI Kalimantan, serta klaster api eksternal yang tidak berkaitan (`ignored_by_event` + *buffer* dilasi 1 piksel) ditandai dengan `valid = 0`. Seluruh perhitungan *loss* dan metrik evaluasi wajib di-*mask* hanya pada `valid == 1`.
4. **Protokol Anti-Kebocoran Kausal (*Strict Anti-Leakage*)**:
   - Lapisan tutupan lahan dan kehilangan hutan tahunan dibekukan pada tahun $Y-1$ untuk memprediksi tahun $Y$.
   - Indeks iklim (`ONI`, `DMI`) dan frekuensi kebakaran historis dihitung secara kausal tanpa *lookahead*.
   - Prakiraan cuaca NWP GFS menggunakan inisialisasi 00:00 UTC pada hari $t$.
   - Pelabelan `y_new` menerapkan ekspansi kausal bertahap ($t \to t+1 \to t+2 \to t+3$, batas $\le 3{,}5\text{ piksel/hari}$) dan pengisian *pinhole* 1-piksel internal ($\ge 5$ tetangga aktif dari 8-ketetanggaan) tanpa mendilasi batas luar api.

---

## 3. Struktur Berkas & Skema Tensor `.npz`

Berkas siap-latih berada di `output/dataset_npz/` yang terdiri atas `train.npz`, `val.npz`, `test.npz`, dan `metadata.json`.

### 3.1 Skema Array pada Setiap Berkas `.npz`

| Kunci Array | Dimensi (*Shape*) | Tipe Data | Deskripsi |
| :--- | :--- | :--- | :--- |
| **`x`** | `[N, 32, 64, 64]` | `float16` | Tumpukan 32 fitur masukan (`Core-32`) ternormalisasi dan dijamin bebas `NaN`. |
| **`y`** *(Target)* | `[N, 3, 64, 64]` | `uint8` | Masker biner area terbakar **kumulatif** untuk horizon $t+1, t+2, t+3$ (`burned_t \| y_new`). |
| **`y_new`** *(Target)* | `[N, 3, 64, 64]` | `uint8` | Masker biner piksel **baru terbakar** secara kausal pada rentang $(t, t+k]$. |
| **`valid`** | `[N, 3, 64, 64]` | `uint8` | Masker validitas piksel (`1` = daratan AOI teramati valid, `0` = awan/air/luar AOI/klaster eksternal). |
| **`state_t`** | `[N, 3, 64, 64]` | `float16` | Kondisi awal untuk `Neural-CA`: `[P_U (belum terbakar), P_F (aktif hari t), P_B (terbakar sebelumnya)]`. |
| **`ca_phys`** | `[N, 5, 64, 64]` | `float16` | Bidang parameter fisika propagasi: `[V_wind, theta_wind, slope/90, theta_upslope, M_fuel]`. |
| **`forcing_g`** | `[N, 3, 4, 64, 64]` | `float16` | Prakiraan cuaca GFS ternormalisasi untuk horizon +24h, +48h, +72h: `[u, v, precip_24h, RH]`. |
| **`scalars`** | `[N, 3]` | `float32` | Skalar telekoneksi iklim dan wilayah: `[ONI, DMI, subregion_id]`. |
| **`bbox`** | `[N, 4]` | `float32` | Kotak batas geografis WGS84 (`EPSG:4326`): `[min_lon, min_lat, max_lon, max_lat]` (derajat). |
| **`center_coord`** | `[N, 2]` | `float32` | Koordinat pusat *patch* $(32, 32)$ dalam WGS84: `[lon, lat]` (derajat). |
| **`event_id`** | `[N]` | `int32` | ID klaster kejadian kebakaran. |
| **`tile_id`** | `[N]` | `int32` | ID *tile* spasial di dalam suatu kejadian kebakaran. |
| **`date`** | `[N]` | `<U10` | Tanggal observasi $t$ berformat ISO 8601 (`YYYY-MM-DD`). |
| **`province`** | `[N]` | `<U25` | Nama provinsi administratif di Kalimantan. |

### 3.2 Penyelarasan Koordinat Geografis (`bbox` ke Peta)
Setiap *patch* $64 \times 64$ ($24\text{ km} \times 24\text{ km}$) diproyeksikan langsung ke koordinat WGS84 (`EPSG:4326`) dengan orientasi baris $0$ di sisi paling utara dan kolom $0$ di sisi paling barat:

```python
extent = [bbox[0], bbox[2], bbox[1], bbox[3]]  # [min_lon, max_lon, min_lat, max_lat]
plt.imshow(y_pred[horizon_idx], extent=extent, origin="upper", cmap="YlOrRd")
```

---

## 4. Deskripsi Fitur Masukan (`Core-32`, `Lite-17`, & Tensor Pendukung)

Seluruh fitur kontinu pada `x` dan `forcing_g` dinormalisasi secara ketat menggunakan statistik (`mean` dan `std`) yang **hanya dihitung dari partisi `train`** (tersimpan di `output/stats/split_prd_stats.json` dan `metadata.json`). Variabel curah hujan (`precip_1d`, `precip_7d`, `precip_30d`, `precip_90d`, `gfs_precip`) ditransformasi `log1p` sebelum standarisasi *z-score*, sedangkan fitur kategorikal (`landcover_idx`, `oil_palm_class`) dipertahankan sebagai indeks kelas diskrit.

> **Subset `Lite-17`**: Untuk eksperimen model ringan, indeks kanal di `x` tersedia pada `metadata["lite_17_channel_indices_in_x"]`, yaitu `[0, 1, 2, 3, 4, 5, 6, 7, 14, 15, 19, 20, 22, 26, 27, 28, 29]`.

### 4.1 Daftar Lengkap 32 Kanal Masukan (`Core-32`)

| Indeks | Grup | Sumber Data | Nama Fitur | Deskripsi & Satuan |
| :---: | :--- | :--- | :--- | :--- |
| 0 | **F** (Histori Api) | NASA VIIRS 375 m | `fire_t` | Masker biner api aktif klaster fokus pada hari $t$ (terpusat di `(32, 32)`). |
| 1 | **F** (Histori Api) | NASA VIIRS 375 m | `fire_t-1` | Masker biner api aktif pada hari $t-1$ (bersih dari klaster eksternal). |
| 2 | **F** (Histori Api) | NASA VIIRS 375 m | `fire_t-2` | Masker biner api aktif pada hari $t-2$ (bersih dari klaster eksternal). |
| 3 | **F** (Histori Api) | NASA VIIRS 375 m | `count_14d` | Akumulasi jumlah deteksi api dalam jendela 14 hari $[t-13, t]$. |
| 4 | **F** (Histori Api) | NASA VNP14A1 / FIRMS | `valid_t` | Masker validitas pengamatan pada hari $t$ (`1` = daratan bebas awan). |
| 5 | **F** (Histori Api) | NASA FIRMS Archive | `fire_freq_prev_years` | Frekuensi historis maksimum deteksi bulanan pada tahun-tahun sebelum $\text{year}(t)$. |
| 6 | **S** (Statis Lahan) | Peta Gambut KLHK | `peat` | Indikator keberadaan lahan gambut (`1` = tanah gambut, `0` = tanah mineral). |
| 7 | **S** (Statis Lahan) | MODIS MCD12Q1 (LC1) | `landcover_idx` | Kelas tutupan lahan IGBP (17 kelas) dibekukan dari tahun $Y-1$. |
| 8 | **S** (Statis Lahan) | Global Oil Palm Layer | `oil_palm_class` | Klasifikasi perkebunan kelapa sawit (non-sawit, industri, rakyat). |
| 9 | **S** (Statis Lahan) | Hansen GFC | `treecover_2000` | Persentase tutupan kanopi pohon dasar pada tahun 2000 ($0\text{–}100\%$). |
| 10 | **S** (Statis Lahan) | Hansen GFC | `forest_loss_cum` | Indikator akumulasi kehilangan tutupan hutan hingga tahun $Y-1$. |
| 11 | **S** (Statis Lahan) | OpenStreetMap (OSM) | `dist_road` | Jarak Euclidean ke jaringan jalan terdekat (meter, dibatasi maks. 50 km). |
| 12 | **S** (Statis Lahan) | OpenStreetMap (OSM) | `dist_waterway` | Jarak Euclidean ke sungai/kanal terdekat (meter, dibatasi maks. 50 km). |
| 13 | **S** (Statis Lahan) | OpenStreetMap (OSM) | `dist_settlement` | Jarak Euclidean ke permukiman terdekat (meter, dibatasi maks. 50 km). |
| 14 | **S** (Statis Lahan) | NASA SRTM / NASADEM | `elevation` | Elevasi permukaan digital di atas permukaan laut (meter). |
| 15 | **S** (Statis Lahan) | NASA SRTM / NASADEM | `slope` | Kemiringan lereng medan dari gradien spasial 2D (derajat). |
| 16 | **S** (Statis Lahan) | NASA SRTM / NASADEM | `aspect_sin` | Sinus arah hadap lereng ($\sin(\theta_{\text{aspect}})$). |
| 17 | **S** (Statis Lahan) | NASA SRTM / NASADEM | `aspect_cos` | Kosinus arah hadap lereng ($\cos(\theta_{\text{aspect}})$). |
| 18 | **D** (Iklim Anteseden) | NASA GPM IMERG v07 | `precip_1d` | Akumulasi curah hujan 24 jam pada hari $t$ (mm, `log1p`). |
| 19 | **D** (Iklim Anteseden) | NASA GPM IMERG v07 | `precip_7d` | Akumulasi curah hujan 7 hari anteseden (mm, `log1p`). |
| 20 | **D** (Iklim Anteseden) | NASA GPM IMERG v07 | `precip_30d` | Akumulasi curah hujan 30 hari anteseden (mm, `log1p`). |
| 21 | **D** (Iklim Anteseden) | NASA GPM IMERG v07 | `precip_90d` | Akumulasi curah hujan 90 hari anteseden (mm, `log1p`). |
| 22 | **D** (Iklim Anteseden) | Copernicus CEMS-Fire | `DC` | *Drought Code* (indikator kekeringan lapisan organik dalam). |
| 23 | **D** (Iklim Anteseden) | Copernicus CEMS-Fire | `FFMC` | *Fine Fuel Moisture Code* (kelembaban bahan bakar halus permukaan). |
| 24 | **D** (Iklim Anteseden) | ECMWF ERA5-Land | `RH` | Kelembaban relatif permukaan 2 m ($\%$, aproksimasi August-Roche-Magnus). |
| 25 | **D** (Iklim Anteseden) | ECMWF ERA5-Land | `soil_water_L1` | Kandungan air tanah volumetrik lapisan atas 0–7 cm ($\text{m}^3/\text{m}^3$). |
| 26 | **D** (Iklim Anteseden) | ECMWF ERA5-Land | `wind_u` | Komponen angin zonal (timur–barat) ketinggian 10 m (m/s). |
| 27 | **D** (Iklim Anteseden) | ECMWF ERA5-Land | `wind_v` | Komponen angin meridional (utara–selatan) ketinggian 10 m (m/s). |
| 28 | **G** (Prakiraan NWP) | NOAA NCEP GFS 0.25° | `gfs_wind_u` | Prakiraan komponen angin zonal 10 m horizon 0–24 jam (m/s). |
| 29 | **G** (Prakiraan NWP) | NOAA NCEP GFS 0.25° | `gfs_wind_v` | Prakiraan komponen angin meridional 10 m horizon 0–24 jam (m/s). |
| 30 | **G** (Prakiraan NWP) | NOAA NCEP GFS 0.25° | `gfs_precip` | Prakiraan akumulasi curah hujan 24 jam horizon 0–24 jam (mm, `log1p`). |
| 31 | **G** (Prakiraan NWP) | NOAA NCEP GFS 0.25° | `gfs_RH` | Prakiraan kelembaban relatif 2 m horizon 0–24 jam ($\%$). |

---

## 5. Metodologi Splitting & Protokol Validasi

### 5.1 Metode Utama: `hybrid_tile_disjoint` (Default)

Pembagian *naive* per `event_id` menyebabkan fenomena *event drift* karena sejumlah kejadian kebakaran raksasa (seperti klaster kebakaran 2019 yang masing-masing berada di satu provinsi tunggal) memonopoli partisi `val` atau `test`, sehingga distribusi provinsi dan laju piksel terbakar antar-partisi menjadi timpang dan *validation loss* cenderung stagnan.

Metode **`hybrid_tile_disjoint`** pada `export_npz.py` mengatasi masalah ini dengan tetap menjaga **nol kebocoran temporal (*zero temporal leakage*)**:

1. **Partisi Berbasis `(event_id, tile_id)` untuk Mega-Event**:  
   Kejadian kebakaran berskala masif ($\ge 100$ sampel dan $\ge 10$ *tile*, yaitu `event_id` `10714`, `10802`, `11837`, `13802`, `29054`, dan `29728`) dipartisi pada tingkat **`(event_id, tile_id)`**. Seluruh deret waktu hari berturut-turut ($t, t+1, \dots$) pada satu *tile* tetap berada di satu partisi yang sama, sementara *tile-tile* di dalam mega-event tersebut dibagi secara proporsional (~70% / 15% / 15%) ke `train`, `val`, dan `test`.
2. **Partisi Utuh Berbasis `event_id` untuk Event Kecil–Menengah**:  
   Kejadian di bawah ambang batas di atas tetap dijaga utuh pada level `event_id`.
3. **Optimasi *Multi-Objective Simulated Annealing***:  
   Penempatan kelompok dioptimalkan secara bersamaan untuk menyeimbangkan:
   - Proporsi jumlah sampel (70% : 15% : 15%)
   - Intensitas total piksel api aktif hari berikutnya (`fire_daily[t+1]`)
   - Proporsi 5 provinsi di Kalimantan
   - Proporsi tahun (2017–2026) dan bulan/musim kemarau–basah
   - Keterwakilan porsi mega-event di setiap partisi

#### Statistik Kualitas Partisi Aktual (`seed=42`)

| Partisi | Jumlah Sampel | Jumlah Event | `y` Positif (Kumulatif) | `y_new` Positif (Baru) | Cakupan Piksel Valid |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **`train`** | 3.699 (70,0%) | 219 | 3,233% | 0,414% | 75,5% |
| **`val`** | 790 (15,0%) | 135 | 3,557% | 0,380% | 73,0% |
| **`test`** | 794 (15,0%) | 131 | 3,210% | 0,385% | 75,8% |

**Distribusi Provinsi per Partisi:**
- **Kalimantan Barat**: `train` = 1.166 | `val` = 246 | `test` = 253
- **Kalimantan Selatan**: `train` = 799 | `val` = 169 | `test` = 164
- **Kalimantan Tengah**: `train` = 1.350 | `val` = 290 | `test` = 296
- **Kalimantan Timur**: `train` = 284 | `val` = 63 | `test` = 61
- **Kalimantan Utara**: `train` = 100 | `val` = 22 | `test` = 20

**Verifikasi Anti-Kebocoran & Keseimbangan Distribusi (Jensen-Shannon Divergence, $0 =$ identik):**
- **Overlap pasangan `(event_id, tile_id)` antar partisi**: `train ∩ val = 0`, `train ∩ test = 0`, `val ∩ test = 0`.
- **JSD Provinsi**: `0,000027` (`train` vs `val`) | `0,000095` (`train` vs `test`).
- **JSD Tahun**: `0,000263` (`train` vs `val`) | `0,000236` (`train` vs `test`).
- **JSD Bulan**: `0,004148` (`train` vs `val`) | `0,002771` (`train` vs `test`).
- **JSD Porsi Mega-Event**: `0,000037` (`train` vs `val`) | `0,000053` (`train` vs `test`).
- **Drift Fitur `Core-32`**: selisih rata-rata ternormalisasi maksimum ($\max |\Delta z|$) terhadap `train` hanya `0,163` (`val`) dan `0,141` (`test`).

---

### 5.2 Strategi Partisi Alternatif & Validasi Silang

Skrip `export_npz.py` tetap menyediakan tiga strategi alternatif sesuai kebutuhan eksperimen:

1. **`--strategy balanced_event_disjoint`**:  
   Seluruh `event_id` dijaga utuh (tidak ada mega-event yang dipecah per *tile*), dengan optimasi *simulated annealing* yang membatasi porsi maksimum satu event di `val`/`test` (`--max-val-test-event-frac`, default `0.20`) dan menyeimbangkan provinsi serta tahun.
2. **`--strategy event_disjoint`**:  
   Pengelompokan *greedy* berbasis `event_id` dengan stratifikasi tahunan (`--no-year-stratification` untuk menonaktifkan).
3. **`--strategy chronological`**:  
   Pemisahan maju berbasis batas tahun (default: `train` $\le 2022$, `val` $= 2023$, `test` $= 2024\text{–}2026$) untuk mengevaluasi generalisasi *out-of-time* tanpa *future lookahead*.
4. **Leave-One-Year-Out (LOYO) Cross-Validation**:  
   Tersedia pada `output/manifest.parquet` melalui 10 kolom (`fold_loyo_2017` hingga `fold_loyo_2026`), di mana setiap tahun $Y$ diisolasi sebagai `test`, tahun $Y-1$ sebagai `val`, dan tahun lainnya sebagai `train`.

---

## 6. Panduan Penggunaan Cepat

### 6.1 Mengekspor Ulang Dataset `.npz`

```powershell
$env:PYTHONPATH="src"
.\.venv\Scripts\python.exe export_npz.py --strategy hybrid_tile_disjoint --train-ratio 0.70 --val-ratio 0.15 --test-ratio 0.15 --seed 42
```

### 6.2 Memuat Data di Python

```python
import json
import numpy as np

meta = json.loads(open("output/dataset_npz/metadata.json", encoding="utf-8").read())
train = np.load("output/dataset_npz/train.npz")

x = train["x"].astype(np.float32)          # [3699, 32, 64, 64]
y = train["y"]                              # [3699, 3, 64, 64] (kumulatif)
y_new = train["y_new"]                      # [3699, 3, 64, 64] (baru terbakar)
valid = train["valid"] > 0                  # [3699, 3, 64, 64] (masker validitas)
state_t = train["state_t"].astype(np.float32)
ca_phys = train["ca_phys"].astype(np.float32)
forcing_g = train["forcing_g"].astype(np.float32)
scalars = train["scalars"]                  # [3699, 3] -> [ONI, DMI, subregion_id]
```

