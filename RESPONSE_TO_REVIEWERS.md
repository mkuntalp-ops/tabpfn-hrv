# Response to Reviewers — Draft Template

> **Kullanım:** Dergiden gelen gerçek hakem yorumları elimizde olmadığından,
> bu belge red kararında tipik olarak belirtilen itirazlara göre
> yapılandırılmış bir taslaktır. Her hakem yorumunu numaralandırıp ilgili
> cevabı buraya kopyalayın; `[YENİ DEĞER]` yer tutucularını
> `reproduce_all_results.sh` çıktısındaki gerçek sayılarla doldurun.

---

**Sayın Editör ve Hakemler,**

Hakem değerlendirmeleri için teşekkür ederiz. Tüm eleştiriler üzerine
analiz kodu tamamen gözden geçirilmiş, tespit edilen metodolojik hatalar
düzeltimiştir. Düzeltmelerin tamamlanabilir ve doğrulanabilir olması için
analiz kodu açık erişimli repository'ye taşınmış ve otomatik doğrulama
(CI) altyapısı eklenmiştir. Aşağıda her düzeltme ve makale üzerindeki
etkisi ayrıntılı olarak sunulmaktadır.

## 1. Ön işlem (preprocessing) veri sızıntısı — düzeltildi

**Tespit:** Orijinal analizde eksik değer doldurma (`SimpleImputer`) ve
standartlaştırma (`StandardScaler`), tüm veri kümesi (test fold'ları ve
dış test kohortları dahil) üzerinde uyumlanmıştı. Bu, test verisinin
dağılım bilgisini eğitim aşamasına sızdırmaktaydı.

**Düzeltme:** Ön işlem adımları artık her çapraz doğrulama fold'unda
yalnızca eğitim verisine, LODO analizinde ise yalnızca kaynak kohortlara
uyumlanmaktadır.

**Etki:** Tüm raporlanan metrikler yeniden üretilmiştir. Düzeltilmiş
değerler (leakage içermeyen): ROC-AUC `[YENİ DEĞER]` (önceki:
`[ESKİ DEĞER]`). Beklendiği gibi, nokta tahminleri daha muhafazakâr hale
gelmiştir.

## 2. Hasta seviyesi tanı kuralı — düzeltildi

**Tespit:** Orijinal çalışmada OSA pozitifliği "gecede ≥5 apne dakikası"
ile tanımlanmıştı. Bu kural kayıt süresine göre normalize edilmemiş
olup AHI≥5 klinik eşiği ile eşdeğer değildir (kayıt süreleri hastalar
arasında farklıdır).

**Düzeltme:** Kural, süre-normalize apne yüküne (≥5 apne dakikası/kayıt
saati; AHI≥5 eşleniği) dönüştürülmüştür. Ayrıca Youden eşiği, orijinalde
aynı 35 hasta üzerinde hem optimize hem değerlendirilmekteydi; eşikler
artık leave-one-subject-out ile seçilmektedir.

**Etki:** Hasta seviyesi duyarlılık/özgüllük değerleri değişmiştir:
`[YENİ DEĞER]` (önceki: `[ESKİ DEĞER]`). Yeni değerler iyimserlik iç
etmemektedir.

## 3. Çoklu karşılaştırma düzeltmesi eksikliği — giderildi

**Düzeltme:** Model karşılaştırmaları artık aynı hasta-kümeli bootstrap
resample'ları üzerinde paired olarak yapılmakta ve Holm step-down
düzeltmesi uygulanmaktadır. Sonuçlar Holm düzeltmeli p değerleriyle
raporlanmıştır (Tablo 5).

## 4. Ölü analiz kodu ve tekrarlanabilirlik — düzeltildi

**Tespit:** Repository'de temel model değerlendirme modülleri ve
konfigürasyon dosyaları eksikti; bir analiz script'indeki koşulsuz
`return`, 3×3 transfer matrisi ve grafiklerin çalışmasını engelliyordu
(sonuçlar izlenemez kaynaklardan geliyordu).

**Düzeltme:** Tüm kaynak modüller (`models/`, `evaluation/`,
`configs/`) tamamlanmış, ölü kod kaldırılmış, tek komutla tam yeniden
üretim (`reproduce_all_results.sh`) ve otomatik CI doğrulaması
eklenmiştir. Feature listesi, veri şemasına karşı runtime'da
doğrulanmaktadır.

## 5. İstatistiksel güç ve iddiaların çerçevesi — dürüstleştirildi

**Tespit:** Dış test kümelerindeki few-shot gelişme iddiaları, hasta
seviyesinde kümelenmiş belirsizlik hesaba katıldığında desteklenmiyordu
(her paired delta CI'ı sıfırı kapsamaktadır; UCDDB 6, SLPDB 1 held-out
hasta).

**Düzeltme:** Çalışmanın iddiaları "exploratory / hypothesis-generating"
çerçevesine çekilmiştir. Few-shot karşılaştırmaları, point tahminleri
kümelenmiş %95 CI'larıyla birlikte raporlanmaktadır; hiçbir few-shot
iyileşmesi istatistiksel olarak anlamlı bulunmamıştır. Limitations
bölümü bu güç sınırlamalarını açıkça belirtmektedir.

## 6. ECE hesaplama hatası — düzeltildi

Olasılığı tam 1.0 olan tahminler son kalibrasyon binine dahil
edilmemekteydi; düzeltme tüm ECE değerlerini hafifçe etkilemiştir.

---

## Eklenen materyaller

- Tam ve doğrulanmış analiz kodu (CI ile otomatik test: derleme, import,
  metrik kenar durumları, Holm mantığı)
- `reproduce_all_results.sh` — tek komutla tam yeniden üretim
- Hasta-kümeli bootstrap CI'ları içeren güncellenmiş tüm tablolar
- Holm düzeltmeli çoklu karşılaştırma tablosu
- Yeniden üretilebilirlik beyanı: seed, feature listesi, checkpoint
  sürümü `configs/protocol.py` içinde sabitlenmiştir

Saygılarımızla,
[Yazarlar]
