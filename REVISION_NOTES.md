# Revision Notes — Converting the Manuscript to a Letter / Short Communication

Desk-reject (editör değerlendirmesi, hakem süreci olmadan) tipik olarak
"etki/katkı ile format uyuşmuyor" mesajıdır. Bu çalışmanın verisi tam
makale iddialarını taşımıyor (bkz. aşağıdaki güç tablosu), ancak dürüst
istatistiğiyle **letter / short communication** formatına çok iyi oturur.

## Neden letter

| Tam makale gereksinimi | Çalışmanın durumu |
|---|---|
| Güçlü dış doğrulama | UCDDB: 6 held-out hasta; SLPDB: 1 hasta (CI hesaplanamaz) |
| Anlamlı ana sonuç | Tüm few-shot vs zero-shot deltalarının CI'ları sıfırı kapsıyor |
| Yöntemsel novelti | Klasik HRV öznitelikleri + bilinen veri seti + mevcut modele (TabPFN) uygulama |
| Çoklu kohort iddiası | Fiilen tek güçlü kohort (Apnea-ECG, 33 dış hasta) |

Letter formatının çalışmaya katkısı:
- Tek ana soru taşınır: *"TabPFN, tek kanallı ECG-HRV'den apne tespitinde
  az etiketli ve kohortlar arası geçiş koşullarında klasik ML ile
  karşılaştırılabilir mi?"*
- Hasta-kümeli CI + Holm düzeltmesi letter literatüründe nadir görülen
  bir istatistik sağlamlığıdır → güç tarafı
- "Exploratory" çerçevesi letter'da doğal durur; tam makalede savunulması
  zordu

## Letter yapısı ve kelime bütçesi (hedef ~1200–1500 kelime)

| Bölüm | Bütçe | İçerik |
|---|---|---|
| Başlık | — | "…: a multi-database exploratory evaluation" ifadesi başlıkta |
| Abstract | 100–150 | Tek soru, ana nokta tahminleri (CI ile), non-significant few-shot dürüstçe |
| Giriş | 200–250 | Klinik motivasyon (OSA prevalansı, tek kanallı ECG potansiyeli), boşluk (az etiket/dış kohort), amaç |
| Yöntem | 300–400 | Veri setleri (3 kohort), 9 HRV özniteliği, GroupKFold + LODO, modeller (seed'ler sabit), ön işlem leakage yasağı, hasta-kümeli bootstrap + Holm. Kod erişim cümlesi (GitHub repo) |
| Sonuçlar | 300–400 | Tablo 1 (LODO, CI'lı) ana sonuç; development GroupKFold tek cümle; few-shot "exploratory, non-significant" |
| Tartışma | 250–350 | Ana bulgu (karşılaştırılabilirlik), güç sınırları (6/1 hasta), klinik yorum, gelecek çalışma |
| Tablolar/Figürler | 1–2 tablo | generate_manuscript_tables.py çıktılarından seçim (aşağıda) |

## Hangi tablolar letter'a girer

**Girsin (çekirdek):**
- `table3_lodo_clustered.md` → "Tablo 1: Üç kohortta LODO performansı (hasta-kümeli %95 CI)"
- `table5_holm_comparisons.md` → "Tablo 2: Holm düzeltmeli model karşılaştırmaları"

**Tek cümlede özetlensin:**
- `table1_development_cv` → "Development kohortunda hasta-bağımsız 5-fold CV'de tüm modeller benzer ROC-AUC gösterdi (Tablo S1)."
- `table2_subject_level` → supplement'e (varsa) veya cümleyle

**Exploratory cümleyle geçilsin:**
- `table4_paired_few_shot` → "Few-shot bağlam uyarlama, kümelenmiş CI'lar altında hiçbir kohortta anlamlı iyileşme göstermedi (tüm Δ ROC-AUC CI'ları sıfırı kapsıyor)."

Supplement yoksa tablo numaralarını buna göre kaydırın.

## Doldurulacak yer tutucular

`reproduce_all_results.sh` çalıştırıldıktan sonra:
1. `results/manuscript_tables/table*.csv|md` içindeki gerçek sayılar
2. Yöntem bölümü: 9 özniteliğin listesi (`configs/protocol.py::PRIMARY_FEATURES`)
3. Kod erişim cümlesi: repo URL + "otomatik CI doğrulaması" ifadesi

## Hedef dergiler (letter/short communication kabul eden)

- Physiological Measurement (short communication) — profil birebir
- Sleep and Breathing / Sleep Medicine (letter / short communication)
- Journal of Clinical Monitoring and Computing (letter / technical note)
- Chest / Annals of Thoracic Medicine (research letter, ~1200 kelime)

## Gönderim öncesi kontrol listesi

- [ ] Tüm sayıları yeniden üret: `bash reproduce_all_results.sh`
- [ ] `configs/protocol.py::PRIMARY_FEATURES` veri şemasıyla doğrula
  (script uyuşmazlıkta otomatik durur)
- [ ] Letter taslağını `LETTER_DRAFT.md` iskeletinden yaz
- [ ] Tabloları `results/manuscript_tables/` çıktısından seç
- [ ] Dergi kelime/tablo limitlerine göre kırp
- [ ] Cover letter'da: tek kanallı ECG'nin klinik değeri, veri erişilebilirliği,
  istatistik sağlamlığı (küme CI + Holm) vurgusu
- [ ] Daha önce desk-reject edilen dergiye gitmiyorsanız format değişikliğini
  cover letter'da gerekçelendirme (yeni analiz + kapsam daraltma)

## Geçmiş notlar

Bu çalışmanın önceki tam makale versiyonunda tespit edilen ve düzeltilen
teknik sorunların tam listesi (leakage, hasta-seviyesi kural, Youden
optimizmi, ECE kenar durumu, ölü kod, çoklu karşılaştırma) analiz kodunda
giderilmiştir; letter bu düzeltilmiş pipeline'ın çıktılarını kullanır.
Olası eski-yeni değer karşılaştırmaları gerekirse `metrics_errata_log.md`
uzatılabilir.
