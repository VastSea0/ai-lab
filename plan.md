# Görsel Sinir Ağı Sandbox Geliştirme Planı

## Özet
Amaç, mevcut sandbox’ı “profesyonel ML/DL öğrenme laboratuvarı” haline getirmek: kullanıcı her epoch, sample, forward/backward adımı, nöron denklemi, ağırlık güncellemesi ve sonuç değişimini yakından inceleyebilecek. Uygulama modüler kalacak; yeni veri setleri, model presetleri, dersler ve görselleştirme modları registry/komponent ekleyerek büyütülecek.

## Ana Değişiklikler
- `SandboxApp` sadeleştirilecek; büyük UI parçaları ayrı modüllere bölünecek: ağ canvas’ı, sol mimari paneli, sağ denetçi, eğitim konsolu, dataset studio, yakın bakış modalları.
- Lab state tek merkezden yönetilecek: seçili görev, veri seti, ağ mimarisi, epoch geçmişi, seçili sample/step, playback durumu, açıklama modu ve görselleştirme modu.
- Saf TypeScript ML motoru korunacak; PyTorch/TensorFlow eklenmeyecek. Forward pass, loss, backprop, gradient ve weight update hesapları mevcut `lib/ml` katmanında genişletilecek.
- UI daha ferah hale getirilecek: sıkışık paneller yerine tab’li/drawer/modal yapıları, net boşluklar, sabit yükseklikli grafikler ve odaklı inceleme ekranları kullanılacak.

## Özellik Planı
- **Hesaplama Playback Sistemi:** Her epoch içindeki sample ve calculation step’ler play/pause, önceki/sonraki adım, hız kontrolü ile izlenecek. Seçili step canvas üzerinde ilgili nöron/bağlantıyı otomatik vurgulayacak.
- **Nöron Yakınlaştırma Modalı:** Nörona tıklanınca büyük bir detay ekranı açılacak; `Σ(x*w)+b`, aktivasyon sonucu, türev, delta, bias ve katkı tablosu gösterilecek.
- **Bağlantı/Ağırlık Modalı:** Bağlantıya tıklanınca input aktivasyonu, weight, gradient, learning rate, update miktarı ve yeni weight adım adım gösterilecek.
- **Gradient Heatmap:** Canvas’ta `Ağırlık`, `Gradient`, `Düzeltme Etkisi` görselleştirme modları olacak. Çizgi rengi/kalınlığı seçili metriğe göre değişecek.
- **Dataset Studio:** CSV/JSON yükleme, tablo önizleme, kolon eşleme, input/target seçimi, normalize etme, train/test split ve hatalı satır uyarıları eklenecek.
- **Model Preset Sistemi:** Regresyon, XOR, daire, spiral ve görüntü-benzeri görevler için hazır mimari presetleri olacak. Preset uygulamak ağı ve geçmişi kontrollü biçimde sıfırlayacak.
- **Açıklama Modları:** `Başlangıç`, `Matematik`, `Mühendis` modları eklenecek. Aynı hesaplama farklı detay seviyelerinde açıklanacak.
- **Aktivasyon Görseli:** Seçili nöronda sigmoid/tanh/relu/linear eğrisi, mevcut `z`, aktivasyon sonucu ve türev noktası mini grafikle gösterilecek.
- **Learning Rate Deneyi:** Mevcut ağı kopyalayıp farklı learning rate değerleriyle kısa simülasyon yapılacak; loss’un nasıl değişeceği asıl modeli bozmadan gösterilecek.
- **Ders Akışı:** Göreve bağlı mini curriculum eklenecek: veri yükle, forward incele, loss’u anla, backprop izle, learning rate dene gibi tamamlanabilir adımlar.

## Veri Modeli ve Arayüz Kararları
- Yeni lab state tipleri eklenecek: `PlaybackState`, `ConceptMode`, `VisualizationMode`, `DatasetMetadata`, `ColumnMapping`, `LessonProgress`.
- `EpochTraceRecord` ve `CalculationStep` genişletilecek; her step hedef nöron/edge id’si, denklem parçaları, önce/sonra değerleri ve açıklama metni taşıyacak.
- Dataset import sonucu sadece client-side tutulacak; v1’de backend, auth veya kalıcı veritabanı eklenmeyecek.
- Preset, lesson ve task tanımları registry mantığıyla tutulacak; yeni konsept eklemek mevcut UI’ı kırmadan mümkün olacak.

## Uygulama Sırası
1. `plan.md` oluştur ve mevcut planı kaydet.
2. Büyük `SandboxApp` bileşenini kontrollü şekilde modüler parçalara ayır.
3. Playback state, step navigation ve canvas step highlight davranışını ekle.
4. Nöron ve bağlantı yakınlaştırma modallarını geliştir.
5. Gradient heatmap ve aktivasyon grafiğini ekle.
6. Dataset Studio’yu mevcut import akışıyla uyumlu şekilde yerleştir.
7. Preset registry, açıklama modları ve ders akışını bağla.
8. Learning rate deney panelini clone/simülasyon mantığıyla ekle.
9. Tip kontrolü, lint, build ve tarayıcı smoke testleriyle doğrula.

## Test Planı
- `npm run typecheck`
- `npm run lint`
- `npm run build`
- Saf ML testleri için Vitest eklenecek: forward sonucu, backprop gradient/update, loss düşüşü, clone simülasyonunun gerçek modeli değiştirmemesi.
- Dataset parser testleri: CSV, JSON, kolon eşleme, normalize, hatalı satır.
- Trace testleri: step sırası, denklem verileri, selected neuron/edge referansları.
- Browser smoke: panel tab’leri, dataset yükleme akışı, epoch playback, modal açma, gradient mode değiştirme.

## Varsayımlar
- Tüm özellikler mevcut Next.js/Tailwind/React yapısı içinde geliştirilecek.
- ML matematiği tamamen TypeScript/JavaScript kalacak.
- İlk sürümde veri setleri tarayıcı belleğinde kalacak.
- Profesyonel görünüm öncelikli olacak; eğitimsel açıklamalar mod/drawer/modal içinde verilecek, ana ekran tekrar sıkıştırılmayacak.
