# Dokumentasi Dataset Penjalaran Kebakaran Hutan dan Lahan Kalimantan (2017–2026)

Dokumentasi ini menyajikan spesifikasi teknis **Kalimantan Wildfire Spread Dataset** yang telah disesuaikan dengan arsitektur *pipeline* pemrosesan multi-satelit serta metodologi partisi terbaru (**`hybrid_tile_disjoint`** dengan rasio **70 : 15 : 15**).

---

## 1. Ringkasan & Cakupan Dataset

**Kalimantan Wildfire Spread Dataset** adalah dataset *spatiotemporal* siap-latih untuk pemodelan prediksi penjalaran muka api kebakaran hutan dan lahan berbasis *deep learning* (misalnya U-Net, SegFormer, Swin-UNet) maupun *Physics-Guided Neural Cellular Automata* (`Neural-CA`). Seluruh lapisan multi-sensor diselaraskan ke grid metrik seragam berukuran $375\text{ m} \times 375\text{ m}$ dan dirangkum menjadi *patch* berpusat pada klaster api aktif.

| Atribut | Spesifikasi |
| :--- | :--- |
| **Wilayah Cakupan** | Seluruh Kalimantan, Indonesia ($4{,}5^\circ\text{LS} - 4{,}5^\circ\text{LU}$, $108{,}5^\circ\text{BT} - 119{,}5^\circ\text{BT}$): Kalimantan Barat, Kalimantan Tengah, Kalimantan Selatan, Kalimantan Timur, Kalimantan Utara |
| **Periode Temporal** | 10 Juli 2017 – 31 Maret 2026 (resolusi harian) |
| **Total Sampel** | **5.283** *patch* kejadian-hari ($64 \times 64$ piksel) |
| **Resolusi Spasial** | $375\text{ m} \times 375\text{ m}$ per piksel ($1\text{ piksel} = 14{,}0625\text{ ha}$; $1\text{ patch} = 24\text{ km} \times 24\text{ km}$) |
| **Sistem Koordinat (CRS)** | Lambert Azimuthal Equal Area (`+proj=laea +lat_0=0 +lon_0=114 +datum=WGS84 +units=m +no_defs`) |
| **Mode ROI** | `active_cluster` — klaster api aktif yang bersebelahan pada hari $t$ ($\ge 5$ piksel) dipusatkan pada koordinat lokal $(32, 32)$ |
| **Horizon Prediksi** | Multi-horizon: $+24\text{ jam}$ ($t+1$), $+48\text{ jam}$ ($t+2$), dan $+72\text{ jam}$ ($t+3$) |
| **Protokol Partisi Default** | `hybrid_tile_disjoint` rasio **70 : 15 : 15** (`train`: **3.699** \| `val`: **790** \| `test`: **794**) |

---

## 2. Karakteristik Masalah & Tantangan Pemodelan

Kebakaran di Kalimantan didominasi oleh kebakaran lahan gambut tropis dan vegetasi terbuka yang intensitasnya meningkat tajam pada fase kering El Niño Southern Oscillation (ENSO) dan Indian Ocean Dipole (IOD) positif (seperti tahun 2019 dan 2023). Dataset ini merepresentasikan empat tantangan utama pemodelan secara realistis:

1. **Ketidakseimbangan Kelas Ekstrem (*Extreme Class Imbalance*)**:  
   Pada gabungan piksel valid `train` + `val`, piksel baru terbakar (`y_new`) hanya mencakup **0,409%** (rasio $\approx 244:1$, yakni $0{,}233\%$ di $t+1$, $0{,}426\%$ di $t+2$, dan $0{,}579\%$ di $t+3$), sedangkan target kumulatif (`y`) mencakup **3,288%** (rasio $\approx 29:1$). Pelatihan disarankan menggunakan fungsi *loss* yang tahan ketidakseimbangan (*masked* Focal/Dice/Tversky loss).
2. ***Zero-Inflation* pada Level *Patch***:  
   Meskipun terdapat api aktif pada hari $t$, sekitar **34,4%** *patch* pada $t+1$, **26,8%** pada $t+2$, dan **24,0%** pada $t+3$ tidak menunjukkan perluasan piksel baru (`y_new = 0`).
3. **Oklusi Awan & Masking Spasial (`valid`)**:  
   Tutupan awan tropis, badan air permanen, area di luar batas AOI Kalimantan, sumber panas industri persisten, serta klaster api eksternal yang tidak berkaitan (`ignored_by_event` + *buffer* dilasi 1 piksel) ditandai dengan `valid = 0`. Seluruh perhitungan *loss* dan metrik evaluasi wajib di-*mask* hanya pada `valid == 1`.
4. **Protokol Anti-Kebocoran Kausal (*Strict Anti-Leakage*)**:  
   Seluruh prediktor dinamis tahunan dibekukan pada tahun $Y-1$, indeks telekoneksi bulanan diambil dari bulan kalender sebelumnya ($m-1$), prakiraan GFS dibatasi pada waktu pembuatan (`creation_time`) sebelum `00:00 UTC` hari $t+1$, dan statistik normalisasi fitur dihitung eksklusif dari partisi `train`.

---

## 3. Sumber Data & Pipeline Preprocessing Multi-Satelit

Dataset ini mengintegrasikan pengamatan termal satelit, lapisan tutupan lahan & infrastruktur, data hidrometeorologi anteseden, serta prakiraan cuaca numerik (NWP). Setiap sumber data melalui tahapan pembersihan, penyelarasan kausal, dan penyamaan resolusi sebelum digabungkan menjadi tensor per kejadian (`event_XXXXX.zarr`) dan diekspor ke `.npz`.

### 3.1 Harmonisasi Grid Spasial & Teknik Resampling

- **Grid Referensi Metrik (`375 m`)**: Seluruh AOI Kalimantan (batas administratif 5 provinsi GADM Level-1) diproyeksikan ke CRS *equal-area* **LAEA** agar setiap piksel memiliki luas konsisten ($375\text{ m} \times 375\text{ m} = 14{,}0625\text{ ha}$) dan jarak metrik antar-piksel seragam untuk operasi konvolusi dan propagasi CA.
- **Resampling dari Grid Kasar ke Patch 375 m**:
  - **Interpolasi Bilinear Separabel (`order=1`)**: Diterapkan pada variabel meteorologi & bahaya kebakaran kontinu (IMERG 10 km, ERA5-Land 10 km, CEMS-Fire 25 km, dan GFS 25 km) untuk mencegah artefak blok grid kasar pada batas *patch* 375 m.
  - **Nearest-Neighbor (`order=0`)**: Diterapkan pada lapisan diskrit/kategorikal seperti masker pengamatan `VNP14A1` (1 km) agar kode kelas kualitas observasi dan badan air tidak terdistorsi oleh interpolasi pecahan.

### 3.2 Ringkasan & Detail Preprocessing per Sumber Data

| Sumber Data | Resolusi Asli → Target | Kanal / Masker yang Dihasilkan | Ringkasan Preprocessing Utama |
| :--- | :--- | :--- | :--- |
| **NASA FIRMS VIIRS 375 m** (`SP` + `NRT`) | $375\text{ m}$ (titik) → $375\text{ m}$ | `fire_t`, `fire_t-1`, `fire_t-2`, `count_14d`, `fire_freq_prev_years`, target `y`, `y_new` | Filter confidence `n`/`h` & `type=0`, deduplikasi koordinat-waktu, filter panas persisten, klasterisasi spasiotemporal 3D, pelabelan ekspansi geodesik |
| **NASA VIIRS `VNP14A1`** (`FireMask`) | $1\text{ km}$ → $375\text{ m}$ | `valid_t`, masker `valid` harian | Reduksi `max` harian di GEE, resampling *nearest-neighbor*, validasi kelas daratan bebas awan `{5..9}` dan eksklusi air (`!= 3`) |
| **NASA SRTM / NASADEM** (`USGS/SRTMGL1_003`) | $30\text{ m}$ → $375\text{ m}$ | `elevation`, `slope`, `aspect_sin`, `aspect_cos`, `theta_upslope` | Proyeksi ulang LAEA, gradien spasial 2D untuk turunan kemiringan lereng ($0\text{–}90^\circ$), komponen trigonometri arah lereng, dan arah tanjakan CA |
| **MODIS `MCD12Q1`** (`LC_Type1` IGBP) | $500\text{ m}$ → $375\text{ m}$ | `landcover_idx`, basis `M_fuel` | Pemilihan tahun kausal $\le Y-1$, pembulatan indeks kelas $[0, 17]$ (tanpa *z-score*), penandaan bahan bakar nol untuk air/perkotaan/es |
| **Hansen GFC** (`v1.13`) | $30\text{ m}$ → $375\text{ m}$ | `treecover_2000`, `forest_loss_cum` | Binarisasi akumulasi tahun kehilangan hutan kausal `1 <= lossyear <= (Y - 2001)` pada grid 375 m |
| **Peta Gambut & Lapisan Sawit Global** | $1\text{ km}$ / $10\text{ m}$ → $375\text{ m}$ | `peat`, `oil_palm_class` | Unmask nol ke grid 375 m; gambut dibinerkan (`1` gambut, `0` mineral); sawit dipertahankan sebagai indeks kelas diskrit $[0, 3]$ |
| **OpenStreetMap (`kalimantan-latest.osm.pbf`)** | Vektor → $375\text{ m}$ | `dist_road`, `dist_waterway`, `dist_settlement` | Rasterisasi garis/poligon/titik (`all_touched=True`) lalu transformasi jarak Euclidean (`EDT` × $375\text{ m}$), di-*clip* pada batas $50\text{ km}$ |
| **NASA GPM IMERG v07** (`IMERG_V07`) | ~11,1 km (grid $10\text{ km}$) → $375\text{ m}$ | `precip_1d`, `precip_7d`, `precip_30d`, `precip_90d` | Integrasi laju setengah-jaman ($\sum \text{rate} \times 0{,}5\text{ jam}$), akumulasi *cumsum* kausal lokal 1/7/30/90 hari, resampling bilinear, `log1p` |
| **ECMWF ERA5-Land Daily** | ~11,1 km (grid $10\text{ km}$) → $375\text{ m}$ | `RH`, `soil_water_L1`, `wind_u`, `wind_v` | Perhitungan analitik `RH` dari suhu $2\text{ m}$ & titik embun $2\text{ m}$ via rumus August-Roche-Magnus, resampling bilinear ke 375 m |
| **Copernicus CEMS-Fire Daily** | ~27,8 km (grid $25\text{ km}$) → $375\text{ m}$ | `DC`, `FFMC` | Agregasi harian `drought_code` dan `fine_fuel_moisture_code`, penandaan hari invalid jika hilang, resampling bilinear ke 375 m |
| **NOAA NCEP GFS 0.25°** | ~27,8 km (grid $25\text{ km}$) → $375\text{ m}$ | `gfs_wind_u`, `gfs_wind_v`, `gfs_precip`, `gfs_RH` + `forcing_g` 3 horizon | Seleksi *run* dengan `creation_time <= 00:00 UTC (t+1)`, rata-rata angin/RH per 24h, penjumlahan tepat 4 langkah 6-jaman untuk curah hujan 24h, `log1p` |
| **NOAA PSL Indeks Iklim** | Skalar bulanan | `scalars[0..1]` (`ONI`, `DMI`) | Pengambilan nilai indeks pada **bulan kalender sebelumnya ($m-1$)** untuk menghindari kebocoran akhir bulan berjalan |

#### Penjelasan Preprocessing Mendalam per Kelompok Sumber Data

1. **Deteksi Titik Panas Aktif & Pembentukan Kejadian (NASA FIRMS VIIRS 375 m)**:
   - **Penggabungan Produk**: Mengambil produk arsip terkalibrasi **`VIIRS_SNPP_SP`** untuk rentang historis dan menyambungnya secara otomatis ke produk **`VIIRS_SNPP_NRT`** mulai $H+1$ setelah tanggal akhir `SP`, diunduh dalam jendela maksimal 5 hari per provinsi.
   - **Pembersihan Kualitas**: Deteksi berkeyakinan rendah (`l` / `low`) dan objek non-vegetasi (`type != 0`) dibuang sehingga hanya menyisakan kepercayaan `nominal` (`n`) dan `high` (`h`). Duplikat berdasarkan `(latitude, longitude, acq_date, acq_time)` dieliminasi sebelum dipetakan ke koordinat sel `(row, col)` grid 375 m.
   - **Penyaringan Sumber Panas Persisten (*Persistent Heat Filter*)**: Sel piksel yang menyala konsisten pada lokasi tetap ($\ge 15$ hari/tahun dan $\ge 9$ bulan berbeda selama $\ge 2$ tahun kalender) diklasifikasikan sebagai anomali termal permanen (suar migas/*gas flaring*, pabrik peleburan, atau kawasan industri), dikeluarkan dari himpunan api aktif, dan di-*mask* permanen pada `aoi_mask` (`valid = 0`).
   - **Klasterisasi & Riwayat Api**: Deteksi non-persisten dikelompokkan menjadi kejadian kebakaran (`event_id`) menggunakan graf *voxel* 3D (`d_max = 10 km`, celah hari `g_max = 2 hari`, disambung lintas batas tahun via *Union-Find*). Kejadian yang lebih besar dari $64 \times 64$ piksel dibagi ke beberapa `tile_id` ber-overlap 25%. Fitur `fire_freq_prev_years` mengambil maksimum deteksi bulanan per piksel hanya dari tahun-tahun sebelum `year(t)` agar kausal.

2. **Masking Observasi Satelit Harian (`NASA/VIIRS/002/VNP14A1`)**:
   - Band `FireMask` diagregasi harian menggunakan `max()` pada grid 1 km, lalu dipetakan ke 375 m via *nearest-neighbor* (`order=0`).
   - Suatu piksel dinyatakan teramati valid (`valid_t = 1`) hanya jika kelas `FireMask ∈ {5, 6, 7, 8, 9}` (piksel daratan bebas awan atau api terdeteksi) atau memiliki deteksi FIRMS terkonfirmasi pada sel tersebut, serta **bukan badan air** (`FireMask != 3` dan `MCD12Q1 != 17`) dan berada di dalam batas daratan AOI (termasuk di luar sel sumber panas persisten). Apabila citra harian `VNP14A1` tidak tersedia, sistem kembali ke masker daratan `aoi_mask & (LC_Type1 != 17)`.

3. **Topografi & Derivasi Parameter Fisika Lereng (NASA SRTM / NASADEM)**:
   - Elevasi 30 m disampel ulang ke grid 375 m, lalu dihitung gradien spasial ke arah timur ($\partial Z / \partial x$) dan utara ($\partial Z / \partial y$) di atas grid metrik untuk menurunkan:
     - `slope` ($^\circ$): $\arctan\left(\sqrt{(\partial Z/\partial x)^2 + (\partial Z/\partial y)^2}\right) \cdot \frac{180}{\pi} \in [0, 90]$.
     - `aspect_sin` dan `aspect_cos`: dihitung dari arah hadap lereng kompas $\theta_{\text{aspect}} = \operatorname{atan2}(-\partial Z/\partial x, -\partial Z/\partial y) \bmod 2\pi$ agar sifat periodik sudut tetap halus dan aman terhadap augmentasi *flip*/rotasi.
     - `theta_upslope`: sudut matematika arah tanjakan tercuram $\operatorname{atan2}(\partial Z/\partial y, \partial Z/\partial x) \bmod 2\pi$ (radian) yang disimpan ke `ca_phys[3]`.

4. **Tutupan Lahan, Gambut, Sawit, & Kehilangan Hutan (Lapisan Biofisik Statis/Tahunan)**:
   - **MODIS `MCD12Q1` (`LC_Type1`)**: Menggunakan tahun rilis $\le Y-1$ untuk seluruh sampel tahun $Y$; dibulatkan dan dibatasi pada rentang indeks $[0, 17]$ agar siap masuk ke *embedding layer*. Kelas `17` (perairan), `13` (perkotaan), dan `15` (salju/es) otomatis mematikan ketersediaan bahan bakar pada kanal fisika `ca_phys[4]` (`M_fuel = 0`).
   - **Hansen GFC**: `treecover2000` merepresentasikan kanopi dasar tahun 2000 ($0\text{–}100\%$), sedangkan `forest_loss_cum` dihitung sebagai masker biner kausal `1 <= lossyear <= (Y - 2001)` sehingga deforestasi setelah tahun $Y-1$ tidak pernah bocor ke data latih.
   - **Gambut & Sawit**: Peta gambut 1 km di-*unmask* menjadi biner (`1` gambut, `0` mineral) dan diselaraskan ke grid 375 m; peta sawit 10 m dikuantisasi ke indeks diskrit $[0, 3]$.

5. **Jarak Infrastruktur Antropogenik (OpenStreetMap)**:
   - Data vektor jalan (`highway IS NOT NULL`), perairan darat (`canal`, `river`, `drain`, `ditch`, `stream`), serta permukiman (`place` titik + poligon `residential`/`building`) diekstrak dari `kalimantan-latest.osm.pbf` menggunakan *driver* GDAL/pyogrio, dirasterisasi dengan opsi `all_touched=True` ke grid 375 m, kemudian dihitung jarak Euclidean terpendek menggunakan `scipy.ndimage.distance_transform_edt` ($\text{piksel} \times 375\text{ m}$) dan dipotong (*clip*) pada batas maksimum $50.000\text{ m}$ ($50\text{ km}$).

6. **Hidrometeorologi Anteseden (NASA GPM IMERG v07, ECMWF ERA5-Land, Copernicus CEMS-Fire)**:
   - **IMERG v07**: Laju curah hujan setengah-jaman ($\text{mm/jam}$) dijumlahkan dan dikalikan $0{,}5\text{ jam}$ per hari pada grid 10 km mulai $t_{\min}-90\text{ hari}$. Akumulasi 1, 7, 30, dan 90 hari diekstrak lewat *prefix cumulative sum* lokal hingga hari $t$, diinterpolasi bilinear ke 375 m, lalu ditransformasi $\log(1 + x)$ sebelum standarisasi.
   - **ERA5-Land**: Rata-rata harian suhu $2\text{ m}$ ($T$) dan titik embun $2\text{ m}$ ($T_d$) dalam Kelvin dikonversi ke Celsius untuk menurunkan kelembaban relatif permukaan (`RH`, $[0, 100]\%$) melalui persamaan **August-Roche-Magnus**:
     $$\text{RH} = \operatorname{clip}\left(100 \cdot \frac{\exp\left(\frac{17{,}625 \cdot T_d}{243{,}04 + T_d}\right)}{\exp\left(\frac{17{,}625 \cdot T}{243{,}04 + T}\right)},\; 0,\; 100\right)$$
     bersama kelembaban tanah `soil_water_L1` serta angin $10\text{ m}$ (`wind_u`, `wind_v`), semuanya diinterpolasi bilinear ke 375 m.
   - **CEMS-Fire**: `DC` dan `FFMC` pada grid 25 km divalidasi ketersediaannya per tanggal lalu diinterpolasi bilinear ke 375 m.

7. **Prakiraan Cuaca NWP & Indeks Telekoneksi (NOAA GFS 0.25° & NOAA PSL)**:
   - **NOAA GFS 0.25°**: Hanya memilih *run* dengan `creation_time` dalam $[T_{\text{cutoff}} - 24\text{h},\; T_{\text{cutoff}}]$ di mana $T_{\text{cutoff}} = \text{00:00 UTC}$ pada hari $t+1$. Untuk 3 horizon (`0–24h`, `24–48h`, `48–72h`), angin (`u`, `v`) dan `RH` dirata-ratakan pada seluruh langkah jam prakiraan di jendela masing-masing. Untuk `total_precipitation_surface` (yang bernilai akumulatif dalam siklus 6 jam), total curah hujan 24 jam dijumlahkan secara eksklusif dari **4 titik batas 6-jaman** (`{6,12,18,24}`, `{30,36,42,48}`, dan `{54,60,66,72}`) agar bebas *double counting*, lalu ditransformasi `log1p`.
   - **ONI & DMI**: Nilai indeks bulanan diambil secara ketat pada **bulan kalender sebelum tanggal sampel ($m-1$)**.

### 3.3 Pembentukan Label Kausal (`y`, `y_new`, `valid`) & Standarisasi Fitur

- **Penyaringan Sampel & Pemusatan Klaster Fokus**: Hanya sampel yang memiliki $\ge 5$ piksel aktif pada hari $t$, terhubung dalam jendela prediksi masa depan, dan memiliki data meteorologi valid yang dirakit. Jendela $64 \times 64$ dipusatkan pada sentroid klaster aktif terbesar/fokus pada koordinat lokal $(32, 32)$.
- **Ekspansi Kausal & Penanganan Kebisingan Label**:
  - *Pinhole filling*: celah 1-piksel internal yang dikelilingi $\ge 5$ dari 8 tetangga aktif (serta lubang tertutup topologis) diisi tanpa mendilasi batas luar api.
  - Ekspansi kausal bertahap ($t \to t+1 \to t+2 \to t+3$) dengan batas langkah maksimal $\le 3{,}5\text{ piksel/hari}$ memisahkan penjalaran api sejati (`y_new`) dari penyalaan baru jarak jauh/palsu (`ignored_by_event`), yang selanjutnya dibersihkan dari evaluasi melalui masker `valid = 0` (termasuk *buffer* dilasi 1 piksel di sekeliling klaster eksternal).
- **Normalisasi Tanpa Kebocoran (*Train-Only Statistics*)**:
  - Kanal kategorikal (`landcover_idx`, `oil_palm_class`) dan biner/trigonometri (`fire_t`, `fire_t-1`, `fire_t-2`, `valid_t`, `aspect_sin`, `aspect_cos`) tidak dikenakan *z-score*.
  - Kanal kontinu (setelah `log1p` khusus 5 variabel curah hujan) dipotong (*winsorized*) ke rentang persentil $[p_{0{,}01}, p_{99{,}9}]$ dan distandarisasi $\frac{x - \mu_{\text{train}}}{\sigma_{\text{train}}}$ menggunakan statistik yang **hanya dihitung dari partisi `train`**, dengan nilai `NaN`/`Inf` dipetakan ke `0.0`.

---

## 4. Struktur Berkas & Skema Tensor `.npz`

Berkas siap-latih berada di `output/dataset_npz/` yang terdiri atas `train.npz`, `val.npz`, `test.npz`, dan `metadata.json`.

### 4.1 Skema Array pada Setiap Berkas `.npz`

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

### 4.2 Penyelarasan Koordinat Geografis (`bbox` ke Peta)

Setiap *patch* $64 \times 64$ ($24\text{ km} \times 24\text{ km}$) diproyeksikan langsung ke koordinat WGS84 (`EPSG:4326`) dengan orientasi baris $0$ di sisi paling utara dan kolom $0$ di sisi paling barat:

```python
extent = [bbox[0], bbox[2], bbox[1], bbox[3]]  # [min_lon, max_lon, min_lat, max_lat]
plt.imshow(y_pred[horizon_idx], extent=extent, origin="upper", cmap="YlOrRd")
```

---

## 5. Deskripsi Fitur Masukan (`Core-32`, `Lite-17`, & Tensor Pendukung)

> **Subset `Lite-17`**: Untuk eksperimen model ringan, indeks kanal di `x` tersedia pada `metadata["lite_17_channel_indices_in_x"]`, yaitu `[0, 1, 2, 3, 4, 5, 6, 7, 14, 15, 19, 20, 22, 26, 27, 28, 29]`.

### 5.1 Daftar Lengkap 32 Kanal Masukan (`Core-32`)

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
| 8 | **S** (Statis Lahan) | Global Oil Palm Layer | `oil_palm_class` | Klasifikasi perkebunan kelapa sawit (`0`: non-sawit, `1`: industri, `2`: rakyat). |
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

## 6. Metodologi Splitting (`hybrid_tile_disjoint`; 70:15:15) & Protokol Validasi

### 6.1 Metode Utama: `hybrid_tile_disjoint` (Rasio 70 : 15 : 15)

Pembagian *naive* pada level `event_id` utuh sangat rentan terhadap *event-size / regime drift*: beberapa kejadian kebakaran raksasa (*mega-event*, terutama pada puncak El Niño 2019 yang terkonsentrasi di satu provinsi tertentu) memiliki ratusan *patch* sehingga dapat memonopoli partisi `val` atau `test`, menimbulkan ketimpangan distribusi provinsi serta laju piksel terbakar, dan berujung pada *validation loss* yang stagnan.

Untuk memecahkan masalah tersebut sekaligus mempertahankan **nol kebocoran temporal (*zero temporal leakage*)**, dataset ini menggunakan metode partisi **`hybrid_tile_disjoint`** (rasio target **70% `train` : 15% `val` : 15% `test`**) yang diimplementasikan di `export_npz.py` melalui tiga mekanisme terpadu:

1. **Granularitas Pengelompokan Hibrida (*Event* vs *Event-Tile*)**:
   - **Mega-Event** ($\ge 100$ sampel dan $\ge 10$ *tile*, yakni `event_id` `10714`, `10802`, `11837`, `13802`, `29054`, dan `29728`): Dipartisi pada level **`(event_id, tile_id)`**. Seluruh deret waktu hari berturut-turut ($t, t+1, \dots$) pada satu *tile* geografis dikunci ke satu partisi yang sama, sementara kumpulan *tile* di dalam mega-event disebar proporsional 70 : 15 : 15 ke `train`, `val`, dan `test`.
   - **Event Kecil–Menengah** ($< 100$ sampel atau $< 10$ *tile*): Tetap dijaga utuh pada level **`event_id`** sehingga tidak ada satu kejadian pun yang terpecah lintas partisi.
2. **Optimasi *Multi-Objective Simulated Annealing* (`n_steps=40.000`, `seed=42`)**:
   Penempatan seluruh grup (`group_key`) ke 3 partisi dioptimalkan secara iteratif untuk meminimumkan fungsi objektif gabungan enam komponen deviasi kuadrat terhadap target rasio $(0{,}70;\; 0{,}15;\; 0{,}15)$:
   $$\mathcal{L}_{\text{split}} = 10\,\mathcal{L}_{\text{sampel}} + 8\,\mathcal{L}_{\text{intensitas api }(t+1)} + 4\,\mathcal{L}_{\text{provinsi}} + 3\,\mathcal{L}_{\text{tahun}} + 2\,\mathcal{L}_{\text{bulan}} + 4\,\mathcal{L}_{\text{mega-event}}$$
   dengan tujuan menjaga keseimbangan simultan pada:
   - **Proporsi jumlah sampel** (70% : 15% : 15%),
   - **Intensitas piksel api aktif hari berikutnya** (`fire_daily[t+1]`) sebagai proksi keparahan target,
   - **Proporsi 5 provinsi** di Kalimantan,
   - **Proporsi tahun (2017–2026)** dan **bulan/musim kemarau–basah**, serta
   - **Keterwakilan porsi mega-event** di setiap partisi.
3. **Jaminan Isolasi Temporal & Non-Overlap**:
   Karena penguncian dilakukan per `(event_id, tile_id)` (atau seluruh `event_id`), urutan observasi harian dari satu lokasi kejadian tidak pernah muncul bersamaan di dua partisi berbeda.

#### Statistik Kualitas Partisi Aktual (`hybrid_tile_disjoint`, 70:15:15, `seed=42`)

| Partisi | Jumlah Sampel (Rasio Aktual) | Jumlah Event | `y` Positif (Kumulatif) | `y_new` Positif (Baru) | Cakupan Piksel Valid |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **`train`** | **3.699** (**70,0%**) | 219 | 3,233% | 0,414% | 75,5% |
| **`val`** | **790** (**15,0%**) | 135 | 3,557% | 0,380% | 73,0% |
| **`test`** | **794** (**15,0%**) | 131 | 3,210% | 0,385% | 75,8% |

**Distribusi Provinsi per Partisi:**
- **Kalimantan Barat**: `train` = 1.166 (70,0%) \| `val` = 246 (14,8%) \| `test` = 253 (15,2%)
- **Kalimantan Selatan**: `train` = 799 (70,6%) \| `val` = 169 (14,9%) \| `test` = 164 (14,5%)
- **Kalimantan Tengah**: `train` = 1.350 (69,7%) \| `val` = 290 (15,0%) \| `test` = 296 (15,3%)
- **Kalimantan Timur**: `train` = 284 (69,6%) \| `val` = 63 (15,4%) \| `test` = 61 (15,0%)
- **Kalimantan Utara**: `train` = 100 (70,4%) \| `val` = 22 (15,5%) \| `test` = 20 (14,1%)

**Verifikasi Anti-Kebocoran & Keseimbangan Distribusi (Jensen-Shannon Divergence, $0 =$ identik):**
- **Overlap pasangan `(event_id, tile_id)` antar partisi**: `train ∩ val = 0`, `train ∩ test = 0`, `val ∩ test = 0`.
- **JSD Provinsi**: `0,000027` (`train` vs `val`) \| `0,000095` (`train` vs `test`).
- **JSD Tahun**: `0,000263` (`train` vs `val`) \| `0,000236` (`train` vs `test`).
- **JSD Bulan**: `0,004148` (`train` vs `val`) \| `0,002771` (`train` vs `test`).
- **JSD Porsi Mega-Event**: `0,000037` (`train` vs `val`) \| `0,000053` (`train` vs `test`).
- **Drift Fitur `Core-32`**: selisih rata-rata ternormalisasi maksimum ($\max |\Delta z|$) terhadap `train` hanya `0,163` (`val`) dan `0,141` (`test`).

---

### 6.2 Strategi Partisi Alternatif & Validasi Silang

Selain **`hybrid_tile_disjoint`** (default), skrip `export_npz.py` juga menyediakan strategi partisi lain sesuai skenario evaluasi:

| Strategi (`--strategy`) | Unit Partisi | Kelebihan Utama | Catatan / Penggunaan |
| :--- | :--- | :--- | :--- |
| **`hybrid_tile_disjoint`** *(Default)* | `(event_id, tile_id)` u/ mega-event; `event_id` u/ lainnya | Distribusi spasial, tahun, musim, dan laju api seimbang (JSD $< 0{,}005$) tanpa kebocoran temporal | Direkomendasikan untuk pelatihan & perbandingan model standar (70 : 15 : 15) |
| **`balanced_event_disjoint`** | `event_id` utuh | Isolasi total level kejadian dengan pembatasan fraksi event maks. (`--max-val-test-event-frac`, default `0.20`) | Dipilih jika evaluasi mengharuskan pemisahan penuh pada level `event_id` |
| **`event_disjoint`** | `event_id` utuh | Alokasi *greedy largest-first* berstratifikasi tahun | *Baseline* pemisahan grup event sederhana |
| **`chronological`** | Tahun kalender | Evaluasi *out-of-time* murni (`train` $\le 2022$, `val` $= 2023$, `test` $= 2024\text{–}2026$) | Simulasi peramalan operasional ke masa depan |
| **LOYO Cross-Validation** | Tahun $Y$ | 10 skema lipatan (`fold_loyo_2017`–`fold_loyo_2026`) di `output/manifest.parquet` | Uji ketahanan antar-tahun dan antar-fase iklim |

---

## 7. Panduan Penggunaan Cepat

### 7.1 Mengekspor Ulang Dataset `.npz`

```powershell
$env:PYTHONPATH="src"
.\.venv\Scripts\python.exe export_npz.py --strategy hybrid_tile_disjoint --train-ratio 0.70 --val-ratio 0.15 --test-ratio 0.15 --seed 42
```

### 7.2 Memuat Data di Python

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
