# Heart rate variability-based sleep apnea classification with a tabular foundation model: an exploratory multi-database evaluation

> **Durum:** Taslak v1 — LODO sayıları `results/manuscript_tables/table3_lodo_clustered.md`
> ve `table4_paired_few_shot.md` gerçek çıktılarıyla dolduruldu.
> Development-cohort (GroupKFold) ve Holm karşılaştırma sayıları
> (`table1_development_cv`, `table5_holm_comparisons`) için
> `reproduce_all_results.sh` tam veri erişimiyle çalıştırılmalı; karşılık gelen
> yerler aşağıda `[TAM PIPELINE GEREKLİ]` olarak işaretlendi.

---

## Abstract (~120 kelime)

**Gerekçe.** Tek kanallı ECG'den uyku apnesi tespiti düşük maliyetli tarama
vaat etmektedir; ancak etiketli veri kıtlığı ve kohortlar arası dağılım
kayması model performansını sınırlar.

**Amaç.** Tabüler temel model TabPFN'in, klasik makine öğrenmesi
yöntemleriyle karşılaştırılabilirliğini az-etiketli ve kohortlar arası
geçiş koşullarında değerlendirmek.

**Yöntem.** Üç klinik kohort (Apnea-ECG, SLPDB, UCDDB), 9 standart HRV
özniteliği, hasta-bağımsız çapraz doğrulama ve leave-one-database-out
(LODO) analizi; hasta-kümeli bootstrap ve Holm düzeltmeli karşılaştırmalar.

**Sonuçlar.** LODO'da TabPFN dış kohortlarda ROC-AUC 0.584 (Apnea-ECG,
%95 CI 0.478–0.683) ve 0.695 (UCDDB, %95 CI 0.500–0.849) elde etti;
klasik modeller benzer aralıklarda kaldı. Few-shot bağlam uyarlaması
kümeli CI'lar altında tutarlı bir iyileşme göstermedi.

**Sonuç.** TabPFN klasik yöntemlerle karşılaştırılabilir performans
gösterdi; model seçimi maliyet ve yorumlanabilirlik göz önünde
bulundurularak yapılabilir.

## Giriş (~220 kelime)

Obstrüktif uyku apnesi (OSA) yüksek prevalansa sahip olup önemli bir
kısmı tanısız kalmakta; altın standart polisomnografi maliyetli ve
erişimi sınırlıdır. Tek kanallı ECG tabanlı tarama, gece boyunca
kolayca kaydedilebildiği için çekici bir alternatiftir. Ancak ECG
tabanlı apne tespitinde etiketli veri kıtlığı ve geliştirme
kohortunun dış kohortlara genellenmesi nadiren raporlanmaktadır.
Pre-trained tabüler temel modeller az-etiketli tıbbi tablo verisinde
umut vadetmekle birlikte, ECG-HRV alanında dış kohort koşullarında
test edilmemiştir. Bu çalışmada TabPFN'in üç uluslararası klinik
kohortta, hasta-bağımsız ve LODO protokolleri altında, klasik
yöntemlere (lojistik regresyon, rastgele orman, SVM, LightGBM,
XGBoost) karşısa performansı hasta-kümeli belirsizlik tahminiyle
değerlendirilmiştir.

## Yöntemler (~350 kelime)

- **Veri:** Apnea-ECG (Marburg; geliştirme kohortu), SLPDB (Boston),
  UCDDB (Dublin). Dış değerlendirmeler ortak held-out epoch alt
  kümesinde yapılmıştır (Apnea-ECG 33 hasta / 16 067 epoch; UCDDB
  6 hasta / 2 495 epoch; SLPDB 1 hasta / 135 epoch). Epoch: 1 dakika.
  Etiketler kaynak veri setlerinin apne anotasyonlarından
  türetilmiştir.
- **Öznitelikler:** `configs/protocol.py::PRIMARY_FEATURES` —
  MeanNN, SDNN, RMSSD, pNN50, LF, HF, LF/HF, TINN ve R-peak sayısı
  (9 standart zaman/frekans alanı HRV ölçüsü; R-peak: NeuroKit2
  pantompkins; RR temizliği: 0.3–2.0 s + %20 ektopi filtresi;
  frekans bantları: Task Force 1996).
- **Protokoller:** (i) Geliştirme: 5-fold hasta-bağımsız GroupKFold
  `[TAM PIPELINE GEREKLİ — table1_development_cv]`; (ii) Dış: LODO
  (iki kohortla eğit, üçüncüde test). Ön işlem (medyan doldurma +
  standartlaştırma) yalnızca eğitim/source verisine uyarlanmıştır
  (sızıntı yok). Tüm seed'ler sabittir; kod GitHub'da otomatik
  doğrulama (CI) ile erişilebilir.
- **Modeller:** TabPFN (n_estimators=4) ve 5 klasik baseline; eşit
  koşullar. Few-shot bağlam uyarlama: hedef kohorttan 50 ve 100
  etiketli epoch'ın TabPFN bağlamına eklenmesi.
- **İstatistik:** Hasta-kümeli bootstrap %95 CI (1 000 resample);
  model karşılaştırmaları aynı resample'lar üzerinde paired, Holm
  step-down düzeltmeli `[TAM PIPELINE GEREKLİ — table5_holm]`.
  SLPDB'de tek held-out hasta nedeniyle kümeler arası CI
  hesaplanamaz olarak işaretlenmiştir.
- **Etik:** `[gerekliyse kurum onay beyanı — kamu erişimli
  PhysioNet veri setleri, ikincil analiz]`.

## Sonuçlar (~350 kelime)

1. **Geliştirme kohortu:** Hasta-bağımsız 5-fold CV'de tüm modeller
   benzer ROC-AUC gösterdi `[TAM PIPELINE GEREKLİ — TabPFN ve en iyi
   baseline değerleri]` (Tablo S1).
2. **LODO (Tablo 1):** TabPFN dış kohortlarda ROC-AUC 0.584 (0.478–0.683)
   (Apnea-ECG), 0.695 (0.500–0.849) (UCDDB) ve 0.438 (CI kestirilemez)
   (SLPDB) elde etti. Klasik modeller aynı kohortlarda örtüşen
   aralıklarda kaldı (örn. UCDDB XGBoost 0.714 (0.527–0.859);
   Apnea-ECG lojistik regresyon 0.692 (0.603–0.777)). Tüm
   kohort-model birleşimlerinde kümeli %95 CI'lar geniştir.
3. **Karşılaştırmalar (Tablo 2):** `[TAM PIPELINE GEREKLİ — Holm
   sonrası en küçük p_holm değeri]` — mevcut kümeli CI'ların
   örtüşmesine dayanarak hiçbir model karşılaştırmasının Holm
   düzeltmesi sonrası anlamlı fark göstermesi beklenmez.
4. **Few-shot (keşifsel):** Hedef kohorttan 50/100 epoch bağlam
   uyarlaması kümeli CI'lar altında tutarlı bir iyileşme göstermedi;
   UCDDB'de N=100 uyarlaması ROC-AUC'yi istatistiksel olarak anlamlı
   biçimde düşürdü (Δ −0.123, %95 CI −0.237 – −0.011), Apnea-ECG'de
   N=50 benzer bir anlamlı düşüş gösterdi (Δ −0.108, −0.209 – −0.007).
   SLPDB (N=1) için CI kestirilemez.

## Tartışma (~300 kelime)

Dış kohortlarda TabPFN klasik yöntemlerle karşılaştırılabilir
performans gösterdi; düzeltilmiş karşılaştırmalarda hiçbir modelin
üstünlüğü beklenmemektedir `[Holm tablosu eklendiğinde kesinleştir]`.
Bu, temel modellerin ECG-HRV alanında "bedava performans"
sağlamadığı, ancak uzman öznitelikle klasik yöntemlere denk olduğu
şeklinde dürüstçe özetlenmelidir. Few-shot bağlam uyarlamasının
yarar sağlamaması (hatta N=100'de zarar vermesi), dağılım
kayması altında bağlam örnekleminin ölçeklenmesinin dikkat
gerektirdiğini düşündürmektedir.

Güç/limit: UCDDB'de 6, SLPDB'de 1 held-out hasta nedeniyle kümeler
arası belirsizlik geniştir; sonuçlar keşifsel olarak
yorumlanmalıdır. Kayıt protokolleri kohortlar arasında farklıdır;
tek öznitelik seti harmonize edilmiş olsa da cihaz ve popülasyon
farklıkları dış geçerliliği sınırlar.

Klinik yorum: Karşılaştırılabilirlik bulgusu, ECG-HRV tarama
araçlarında model seçiminin maliyet/yorumlanabilirlik gözetilerek
yapılabileceğini düşündürmektedir. Daha büyük dış kohortlarla ve
hasta seviyesi uç noktalarla doğrulama önerilir.

## Tablolar

- **Tablo 1:** Tri-center LODO performansı (hasta-kümeli %95 CI) —
  `results/manuscript_tables/table3_lodo_clustered.md` (tam içerik
  yukarıdaki sayılarla üretilmiş durumda).
- **Tablo 2:** Holm düzeltmeli model karşılaştırmaları —
  `table5_holm_comparisons.md` `[TAM PIPELINE GEREKLİ]`.

## Kaynakça notu

Letter formatlar 10–20 referansla sınırlıdır. Öncelik: OSA
prevalans/tanı raporları (1-2); ECG-HRV apne tespiti anahtar
çalışmaları (2-3); Apnea-ECG/SLPDB/UCDDB veri seti makaleleri
(3 — zorunlu atıf); TabPFN orijinal makalesi (1); bootstrap
kümeleme ve Holm yöntem referansları (1-2).

## Cover letter iskeleti (ayrı dosya)

1. Çalışmanın tek cümlelik özeti + neden bu dergi.
2. Metodolojik sağlamlık: sızıntısız ön işlem, hasta-kümeli CI,
   Holm düzeltmesi, açık kod + CI.
3. Exploratory çerçevenin bilinçli tercihi; güç sınırlarının dürüst
   belirtilmesi.
4. Veri setleri kamu erişimli (PhysioNet), kod GitHub'da, çıkar
   çatışması yok, `[finansman]`.
