# LODO Metrik Karşılaştırma Düzeltme Kaydı (LODO Metrics Errata Log)

**Tarih**: 2026-10-06 19:12:00
**Çalışma Dizini**: `results/runs/run_20261006_lodo_provenance_fix/`

---

## 1. Geri Çekilen İddia ve Terminoloji Düzeltmesi

1. **Geri Çekilen İfade**: "Bütün metriklerde maksimum fark $5,55 \times 10^{-17}$" iddiası **resmen geri çekilmiştir**.
   - Gerçek durumda:
     - ROC-AUC için 42 karşılaştırmanın tamamında maksimum mutlak fark: **$5,55 \times 10^{-17}$**
     - PR-AUC için 42 karşılaştırmanın tamamında maksimum mutlak fark: **$8,33 \times 10^{-17}$**
     - Brier Skoru için maksimum mutlak fark: **$3,96 \times 10^{-8}$** (SLPDB Common Held-Out Eval Subset, TabPFN Target-$N=100$).
2. **Terminoloji Değişikliği**:
   - "Birebir özdeş" ifadesi kaldırılmış, yerine **"bildirilen hassasiyette uyumlu"** tanımı kabul edilmiştir.
3. **Model Karşılaştırma Sınırı**:
   - Aşırı genelleştirilmiş determinizm iddiası kaldırılmıştır.
   - Tescillenen ifade: *"İncelenen koşullarda Logistic Regression, Random Forest, SVM ve LightGBM metrik farkları en fazla $8,33 \times 10^{-17}$ düzeyindedir."*

---

## 2. Brier Skoru Hesaplama Kontrolü ve Dtype Analizi

1. **Eski `evaluate_predictions` ve `brier_score_loss` Kodunun Dtype Davranışı**:
   - `src/models/train_and_evaluate.py` içinde Brier skoru `sklearn.metrics.brier_score_loss(y_true, y_prob)` ile hesaplanmaktadır.
   - Scikit-learn kaynak kodunda (`brier_score_loss`):
     ```python
     transformed_labels = xp.astype(transformed_labels, y_proba.dtype, copy=False)
     brier_score = _average(
         xp.sum((transformed_labels - y_proba) ** 2, axis=1), weights=sample_weight
     )
     ```
   - Bu koda göre, `y_proba` girdisinin `dtype` değeri hesaplama hassasiyetini belirler:
     - `y_proba` `float32` ise, etiketler `float32`'ye çevrilir, çıkarma ve kare alma `float32`'de yapılır ve ortalama `float32` olarak toplanır.
     - `y_proba` `float64` ise, tüm işlemler `float64`'te yürütülür.

2. **Mevcut Parquet Tahminleri Üzerinde Yapılan Hesaplama Kontrolü**:
   - Mevcut Parquet dosyasındaki (`lodo_raw_predictions.parquet`) tahminler üzerinde iki yol test edilmiştir:
     - **Yol 1 (Açık float64)**: Olasılıklar çıkarma ve kareden önce float64 olarak tutulup hesaplandığında, eski CSV ile maksimum fark **$3,96 \times 10^{-8}$** olmaktadır.
     - **Yol 2 (Modelin doğal float32 çıktısı yolu)**: Olasılıklar `float32` formatına dönüştürülüp hesaplandığında, TabPFN ve XGBoost koşullarının tamamında (18 koşul) eski CSV değeri ile fark **$\le 5,55 \times 10^{-17}$** (makine tabanında sıfır) olmaktadır.
   - **Tescillenen Sonuç**:
     *"Brier skorları arasında en fazla $3,96 \times 10^{-8}$ mutlak fark saptanmıştır. Bu fark, aynı tahmin vektörü üzerinde metrik hesabının float32 yerine float64 ile yapılmasından kaynaklanmaktadır; sonuçlar bildirilen hassasiyette uyumludur."*

---

## 3. Murphy Ayrışımı Terminolojisi ve Yorumlama Kuralı

1. **Standart Bileşenler**: Murphy (1973) Brier skoru ayrışımının standart bileşenleri strictly:
   $$\text{Brier} = \text{Reliability} - \text{Resolution} + \text{Uncertainty}$$
   (Standart dışı çeviriler olan "kalibrasyon, diskriminasyon ve keskinlik" terimleri kullanılmayacaktır).
2. **Kural Cümlesi**:
   *"Brier, olasılıksal tahminlerin ortalama karesel hatasını ölçer; artışı tek başına kalibrasyon bozulmasını kanıtlamaz."*
3. **UCDDB Güven Aralıkları**:
   - UCDDB ortak test kümesindeki 6 hasta birimli kümelenmiş bootstrap aralıkları keşifsel belirsizlik aralığıdır; klinik veya istatistiksel eşdeğerlik veya kalibrasyon güvencesi olarak yorumlanamaz.
