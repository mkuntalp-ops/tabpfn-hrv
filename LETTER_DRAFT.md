# Research Letter — Taslak İskeleti

> **Kullanım:** Bu iskelet, `reproduce_all_results.sh` +
> `generate_manuscript_tables.py` çıktılarındaki gerçek sayılar
> `[YENİ DEĞER]` yer tutucularına yazılarak doldurulur. Hedef uzunluk:
> ~1200–1500 kelime (derginin limitine göre kırpılır).

---

## Başlık (öneri)

Heart rate variability-based sleep apnea classification with a tabular
foundation model: an exploratory multi-database evaluation

*(Türkçe referans: "Tabüler temel modelle HRV tabanlı uyku apnesi
sınıflandırması: keşfedici çoklu-veritabanı değerlendirmesi" — gönderim
İngilizce yapılacaksa başlık yukarıdaki gibi kullanılmalı.)*

## Abstract (~120 kelime)

**Gerekçe.** Tek kanallı ECG'den uyku apnesi tespiti düşük maliyetli
tarama vaat etmektedir; ancak etiketli veri kıtlığı ve kohortlar arası
dağılım kayması model performansını sınırlar.

**Amaç.** Tabüler temel model TabPFN'in, klasik makine öğrenmesi
yöntemleriyle karşılaştırılabilirliğini az-etiketli ve kohortlar arası
geçiş koşullarında değerlendirmek.

**Yöntem.** Üç klinik kohort (Apnea-ECG, SLPDB, UCDDB), 9 standart HRV
özniteliği, hasta-bağımsız çapraz doğrulama ve leave-one-database-out
(LODO) analizi; hasta-kümeli bootstrap ve Holm düzeltmeli karşılaştırmalar.

**Sonuçlar.** `[YENİ DEĞER — development ROC-AUC CI]`; LODO'da TabPFN
ROC-AUC `[YENİ DEĞER]` (hasta-kümeli %95 CI `[YENİ DEĞER]`); hiçbir model
 karşılaştırması Holm düzeltmesi sonrası anlamlı farklılık göstermedi
(p ≥ `[YENİ DEĞER]`).

**Sonuç.** TabPFN klasik yöntemlerle karşılaştırılabilir performans
göstermiş olup `[klinik yorum — tek cümle]`.

## Giriş (~220 kelime)

Yazım rehberi:
1. **Klinik motivasyon (2-3 cümle):** OSA prevalansı, tanısız kalma oranı,
   PSG'nin maliyeti/erişilebilirliği → tek kanallı ECG taramasının çekiciliği.
2. **Boşluk (2 cümle):** ECG tabanlı apne tespitinde etiketli veri kıtlığı;
   geliştirme kohortunun dış kohortlara genellenmesi nadiren raporlanıyor.
   Temel modellerin (pre-trained tabüler modeller) az-etiketli tıbbi
   tablo verisinde umut vadettiği, ancak ECG-HRV'de dış kohort
   koşullarında test edilmediği.
3. **Amaç (1 cümle):** Bu çalışmada TabPFN'in üç uluslararası klinik
   kohortta, hasta-bağımsız ve LODO protokolleri altında, klasik
   yöntemlere (lojistik regresyon, rastgele orman, SVM, LightGBM,
   XGBoost) karşı performansı hasta-kümeli belirsizlik tahminiyle
   değerlendirilmiştir.

## Yöntemler (~350 kelime)

- **Veri:** Apnea-ECG (Marburg, `[N]` kayıt; geliştirme; c06 hariç),
  SLPDB (Boston, `[N]` kayıt), UCDDB (Dublin, `[N]` kayıt). Epoch: 1 dk.
  Etiketler kaynak veri setlerinin apne annotasyonlarından türetilmiştir.
- **Öznitelikler:** `[configs/protocol.py::PRIMARY_FEATURES listesi]`
  (9 standart zaman/frekans alanı HRV ölçüsü).
- **Protokoller:** (i) Geliştirme: 5-fold hasta-bağımsız GroupKFold;
  (ii) Dış: LODO (iki kohortla eğit, üçüncüde test). Ön işlem (medyan
  doldurma + standartlaştırma) yalnızca eğitim/source verisine
  uyumlanmıştır (sızıntı yok). Tüm seed'ler sabit; kod GitHub'da
  otomatik doğrulama (CI) ile erişilebilir.
- **Modeller:** TabPFN (n_estimators=4) ve 5 klasik baseline; eşit
  koşullar.
- **İstatistik:** Nokta tahminleriyle hasta-kümeli bootstrap %95 CI
  (1.000 resample); model karşılaştırmaları aynı resample'lar üzerinde
  paired, Holm step-down düzeltmeli. SLPDB'de tek held-out hasta nedeniyle
  kümeler arası CI hesaplanamaz olarak işaretlenmiştir.
- **Etik:** `[gerekliyse kurum onayı beyanı]`.

## Sonuçlar (~350 kelime)

**Yapı:**
1. Geliştirme kohortu (kısa): "Hasta-bağımsız 5-fold CV'de tüm modeller
   benzer ROC-AUC gösterdi: TabPFN `[DEĞER]`, en iyi baseline `[DEĞER]`
   (Tablo S1 / cümle içi)."
2. **Ana tablo (Tablo 1):** `results/manuscript_tables/table3_lodo_clustered.md`
   içeriği — kohort başına ROC-AUC (CI), PR-AUC (CI), Brier (CI).
3. Karşılaştırmalar (Tablo 2): `table5_holm_comparisons.md` — "hiçbir
   karşılaştırma Holm sonrası anlamlı değil (en küçük p_holm `[DEĞER]`)."
4. Few-shot (exploratory cümle): "Hedef kohorttan 50/100 epoch bağlam
   uyarlaması, kümelenmiş CI'lar altında hiçbir kohortta anlamlı iyileşme
   göstermedi (tüm Δ CI'ları sıfırı kapsıyor)."

## Tartışma (~300 kelime)

Yazım rehberi:
1. **Ana bulgu (2 cümle):** Dış kohortlarda TabPFN klasik yöntemlerle
   karşılaştırılabilir; hiçbir modelin üstünlüğü düzeltilmiş
   karşılaştırmalarda anlamlı değil. Bu, temel modellerin ECG-HRV
   alanında "bedava performans" sağlamadığı, ancak uzman öznitelikle
   klasik yöntemlere denk olduğu şeklinde dürüstçe özetlenmeli.
2. **Güç/limit (3-4 cümle):** UCDDB 6, SLPDB 1 held-out hasta →
   kümeler arası belirsizlik geniş/keçimsiz; sonuçlar exploratory.
   Tek kayıt cihazı ve öznitelik seti kohortlar arası harmonize edilmiş
   olsa da kayıt protokolleri farklı.
3. **Klinik yorum (2 cümle):** Karşılaştırılabilirlik bulgusu, ECG-HRV
   tarama araçlarında model seçiminin maliyet/interpretabilite
   gözetilerek yapılabileceğini düşündürmektedir.
4. **Gelecek (1 cümle):** Daha büyük dış kohortlarla ve hasta seviyesi
   uç noktalarla doğrulama önerilir.

## Tablolar

- **Tablo 1:** Tri-center LODO performansı (hasta-kümeli %95 CI) —
  `table3_lodo_clustered.md`'den
- **Tablo 2:** Holm düzeltmeli model karşılaştırmaları —
  `table5_holm_comparisons.md`'den

## Kaynakça notu

Letter formatları genelde 10-20 referansla sınırlar. Öncelik:
1. OSA prevalans/tanı raporları (1-2)
2. ECG-HRV apne tespiti anahtar çalışmalar (2-3)
3. Apnea-ECG/SLPDB/UCDDB veri seti makaleleri (3 — zorunlu atıf)
4. TabPFN orijinal makalesi (1)
5. Kullanılan istatistik yöntem referansları (bootstrap kümeleme,
   Holm) (1-2)

## Cover letter iskeleti (ayrı dosya olarak)

1. Tek paragraf: çalışmanın tek cümlelik özeti + neden bu dergi
2. Tek paragraf: metodolojik sağlamlık vurgusu (sızıntısız ön işlem,
   hasta-kümeli CI, Holm düzeltmesi, açık kod + CI)
3. Tek paragraf: exploratory çerçevenin bilinçli tercihi (güç sınırları
   dürüstçe belirtilmiş)
4. Bildirim: veri setleri kamu erişimli (PhysioNet), kod GitHub'da,
   çıkar çatışması yok, `[finansman]`.
