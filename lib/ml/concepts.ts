import type { ActivationName } from "./network";

export type ConceptCategory =
  | "activation"
  | "neuron"
  | "training"
  | "backprop"
  | "data"
  | "task";

export type ConceptId =
  | "linear"
  | "sigmoid"
  | "tanh"
  | "relu"
  | "weight"
  | "bias"
  | "z"
  | "activation"
  | "loss"
  | "gradient"
  | "derivative"
  | "delta"
  | "learning-rate"
  | "epoch"
  | "forward-pass"
  | "backprop"
  | "regression"
  | "classification"
  | "normalization";

export interface ConceptLevelCopy {
  beginner: string;
  math: string;
  engineer: string;
}

export interface Concept {
  id: ConceptId;
  title: string;
  category: ConceptCategory;
  summary: string;
  formula?: string;
  example: string;
  copy: ConceptLevelCopy;
  related: ConceptId[];
}

export const CONCEPTS: Concept[] = [
  {
    id: "linear",
    title: "Linear",
    category: "activation",
    summary: "Gelen değeri değiştirmeden geçirir.",
    formula: "f(z) = z, f'(z) = 1",
    example: "z=0.72 ise çıktı da 0.72 olur.",
    copy: {
      beginner: "Linear aktivasyon nöronun hesapladığı sayıyı aynen dışarı verir. Regresyon çıktılarında kullanışlıdır çünkü çıktı 0-1 arasına sıkışmak zorunda kalmaz.",
      math: "Türevi her yerde 1 olduğu için gradient’i küçültmez veya büyütmez. Bu yüzden çıktı katmanında sürekli değer tahmini için sade bir seçimdir.",
      engineer: "`activationValue('linear', z)` doğrudan `z` döndürür; `activationDerivative` ise 1 döndürür.",
    },
    related: ["activation", "regression"],
  },
  {
    id: "sigmoid",
    title: "Sigmoid",
    category: "activation",
    summary: "Sayıyı 0 ile 1 arasına sıkıştırır.",
    formula: "σ(z)=1/(1+e^-z), σ'(z)=σ(z)(1-σ(z))",
    example: "z çok pozitifse çıktı 1'e, çok negatifse 0'a yaklaşır.",
    copy: {
      beginner: "Sigmoid bir olasılık düğmesi gibi düşünülebilir. Nöronun ne kadar aktif olduğunu 0 ile 1 arasında anlatır.",
      math: "Türevi en büyük z=0 civarındadır. Çok büyük pozitif/negatif z değerlerinde doygunluğa girer ve gradient küçülür.",
      engineer: "Sınıflandırma çıkışlarında okunabilir skor üretir; derin ağlarda doygunluk gradient akışını yavaşlatabilir.",
    },
    related: ["activation", "derivative", "gradient"],
  },
  {
    id: "tanh",
    title: "Tanh",
    category: "activation",
    summary: "Sayıyı -1 ile 1 arasına sıkıştırır.",
    formula: "tanh'(z)=1-tanh(z)^2",
    example: "Negatif sinyali negatif, pozitif sinyali pozitif taşır.",
    copy: {
      beginner: "Tanh, sigmoid’e benzer ama merkez noktası 0’dır. Bu yüzden negatif ve pozitif sinyali daha dengeli gösterebilir.",
      math: "Çıkış aralığı [-1,1] olduğu için hidden katmanlarda ortalaması daha dengeli aktivasyonlar üretebilir.",
      engineer: "Spiral gibi simetrik karar sınırlarında `tanh` hidden katmanlarda iyi bir deneme presetidir.",
    },
    related: ["activation", "derivative"],
  },
  {
    id: "relu",
    title: "ReLU",
    category: "activation",
    summary: "Negatifi 0 yapar, pozitifi aynen geçirir.",
    formula: "ReLU(z)=max(0,z)",
    example: "z=-0.4 ise 0, z=1.2 ise 1.2 üretir.",
    copy: {
      beginner: "ReLU bir kapı gibidir: sinyal pozitifse geçirir, negatifse kapatır.",
      math: "z>0 için türev 1, z<=0 için türev 0 kabul edilir. Kapalı nöron gradient almayabilir.",
      engineer: "Görüntü benzeri görevlerde hidden katmanlarda sık kullanılır; bu lab’da 5x5 çizim görevinde varsayılan presetlerden biridir.",
    },
    related: ["activation", "gradient"],
  },
  {
    id: "weight",
    title: "Ağırlık",
    category: "neuron",
    summary: "Bir bağlantının sinyali ne kadar büyütüp yön değiştireceğini söyler.",
    formula: "katkı = a_önceki × w",
    example: "a=0.8 ve w=-0.5 ise katkı -0.4 olur.",
    copy: {
      beginner: "Ağırlık, iki nöron arasındaki ses düğmesi gibidir. Büyükse sinyal güçlü gider, negatifse yönünü tersine çevirir.",
      math: "Backprop sırasında ∂L/∂w = a_prev × δ_next olarak hesaplanır.",
      engineer: "`EdgeSnapshot` içinde weight, gradient, correction ve weightAfter değerleri gösterilir.",
    },
    related: ["gradient", "learning-rate"],
  },
  {
    id: "bias",
    title: "Bias",
    category: "neuron",
    summary: "Nöronun eşiğini sağa/sola kaydırır.",
    formula: "z = Σ(x*w) + b",
    example: "Gelen katkılar 0 olsa bile bias nöronu aktif hale getirebilir.",
    copy: {
      beginner: "Bias, nörona başlangıç eğilimi verir. Bağlantılardan bağımsız küçük bir ayar kolu gibidir.",
      math: "Bias gradient’i δ değeridir; güncelleme b := b - ηδ olur.",
      engineer: "`NeuronSnapshot.bias` ve `gradientBias` denetçide gösterilir.",
    },
    related: ["z", "delta"],
  },
  {
    id: "z",
    title: "z Toplamı",
    category: "neuron",
    summary: "Aktivasyondan önceki ham nöron toplamıdır.",
    formula: "z = Σ(x*w) + b",
    example: "Gelen tüm katkılar ve bias toplanır, sonra aktivasyona verilir.",
    copy: {
      beginner: "z, nöronun karar vermeden önce topladığı ham puandır.",
      math: "Backprop’ta δ genellikle ∂L/∂z anlamına gelir.",
      engineer: "`NeuronSnapshot.z` canvas ve inspector üzerinde `Σ=` olarak görünür.",
    },
    related: ["activation", "bias", "weight"],
  },
  {
    id: "activation",
    title: "Aktivasyon",
    category: "activation",
    summary: "Nöronun ham toplamını çıktıya çeviren fonksiyondur.",
    formula: "a = f(z)",
    example: "Sigmoid z=0 için a=0.5 üretir.",
    copy: {
      beginner: "Aktivasyon, nöronun topladığı sinyali ne şekilde dışarı vereceğini belirler.",
      math: "Aktivasyonun türevi gradient’in ne kadar geriye geçeceğini belirler.",
      engineer: "Katman ayarındaki activation seçimi tüm layer nöronlarının `activation` alanını değiştirir.",
    },
    related: ["sigmoid", "tanh", "relu", "linear"],
  },
  {
    id: "loss",
    title: "Loss",
    category: "training",
    summary: "Modelin ne kadar yanıldığını tek sayıya indirir.",
    formula: "L = 1/2 × Σ(ŷ-y)^2",
    example: "Tahmin hedefe yaklaştıkça loss küçülür.",
    copy: {
      beginner: "Loss modelin hata puanıdır. Eğitimde amaç bu puanı düşürmektir.",
      math: "Kare hata için ∂L/∂ŷ = ŷ-y olur; bu output delta hesabını başlatır.",
      engineer: "`evaluateLoss` veri setindeki ortalama loss’u hesaplar; history grafiği bunu çizer.",
    },
    related: ["gradient", "backprop"],
  },
  {
    id: "gradient",
    title: "Gradient",
    category: "backprop",
    summary: "Bir değeri değiştirince loss’un hangi yönde değişeceğini söyler.",
    formula: "∂L/∂w = a_prev × δ_next",
    example: "Gradient pozitifse ağırlığı azaltmak loss’u düşürmeye çalışır.",
    copy: {
      beginner: "Gradient, modele hangi düğmeyi hangi yöne çevireceğini söyleyen işarettir.",
      math: "Kısmi türevdir; her parametre için loss eğiminin yerel ölçüsüdür.",
      engineer: "`EdgeSnapshot.gradient` heatmap ve update mikroskobunda kullanılır.",
    },
    related: ["learning-rate", "delta", "derivative"],
  },
  {
    id: "derivative",
    title: "Türev",
    category: "backprop",
    summary: "Fonksiyonun o noktadaki eğimini ölçer.",
    formula: "f'(z)",
    example: "Sigmoid doygunsa türev küçüktür ve gradient zor akar.",
    copy: {
      beginner: "Türev, küçük bir değişikliğin sonucu ne kadar oynatacağını söyler.",
      math: "Zincir kuralında aktivasyon türevi δ hesabına çarpan olarak girer.",
      engineer: "`NeuronSnapshot.derivative` aktivasyon grafiği yanında gösterilir.",
    },
    related: ["activation", "delta"],
  },
  {
    id: "delta",
    title: "Delta",
    category: "backprop",
    summary: "Nöronun loss’a taşıdığı hata sinyalidir.",
    formula: "δ = ∂L/∂z",
    example: "Output nöronunda hata doğrudan tahmin-hedef farkından başlar.",
    copy: {
      beginner: "Delta, hatanın bu nörondan ne kadar geçtiğini anlatır.",
      math: "Hidden katmanda δ = Σ(w_next×δ_next) × f'(z) olur.",
      engineer: "`NeuronSnapshot.delta` backprop animasyonlarında ve focus panelinde gösterilir.",
    },
    related: ["backprop", "derivative", "gradient"],
  },
  {
    id: "learning-rate",
    title: "Learning Rate",
    category: "training",
    summary: "Ağırlıkların her adımda ne kadar değişeceğini belirler.",
    formula: "w_new = w_old - η × gradient",
    example: "η çok büyükse loss zıplayabilir; çok küçükse öğrenme yavaşlar.",
    copy: {
      beginner: "Learning rate adım boyudur. Küçük adım güvenli ama yavaş, büyük adım hızlı ama risklidir.",
      math: "Gradient descent update katsayısı η ile ölçeklenir.",
      engineer: "LR Deneyi paneli modeli klonlayıp farklı η değerlerini gerçek modeli bozmadan simüle eder.",
    },
    related: ["gradient", "loss"],
  },
  {
    id: "epoch",
    title: "Epoch",
    category: "training",
    summary: "Veri setindeki tüm örneklerden bir kez geçmektir.",
    example: "6 örnek varsa 1 epoch, 6 sample update’i anlamına gelir.",
    copy: {
      beginner: "Epoch, modelin tüm veri setini bir tur görmesidir.",
      math: "Her sample loss ve gradient üretir; epoch loss genellikle ortalama loss olarak izlenir.",
      engineer: "`trainEpochDetailed` sample trace’leri ve epoch loss bilgisini döndürür.",
    },
    related: ["forward-pass", "loss"],
  },
  {
    id: "forward-pass",
    title: "Forward Pass",
    category: "training",
    summary: "Girdinin katman katman tahmine dönüşmesidir.",
    formula: "x → z → a → ŷ",
    example: "Piksel değerleri hidden katmanlardan geçip sınıf skorlarına dönüşür.",
    copy: {
      beginner: "Forward, modelin cevap üretme yolculuğudur.",
      math: "Her katmanda matris çarpımı + bias + aktivasyon uygulanır.",
      engineer: "`network.forward` nöronların z, value ve incoming değerlerini doldurur.",
    },
    related: ["activation", "loss"],
  },
  {
    id: "backprop",
    title: "Backpropagation",
    category: "backprop",
    summary: "Hatanın çıktıdan geriye doğru paylaştırılmasıdır.",
    formula: "δ_hidden = Σ(w_next×δ_next) × f'(z)",
    example: "Hata önce output nöronuna, sonra hidden nöronlara geri akar.",
    copy: {
      beginner: "Backprop, modelin hatadan ders çıkarma yoludur.",
      math: "Zincir kuralı ile her parametrenin loss’a etkisi hesaplanır.",
      engineer: "`calculateDeltas` ve `edgeSnapshots` bu lab’daki backprop trace’ini üretir.",
    },
    related: ["delta", "gradient", "derivative"],
  },
  {
    id: "regression",
    title: "Regresyon",
    category: "task",
    summary: "Sürekli bir sayı tahmin etme görevidir.",
    example: "Metrekareden fiyat tahmini gibi.",
    copy: {
      beginner: "Regresyon, modelin bir kategori değil sayı üretmesidir.",
      math: "Genellikle lineer output ve kare hata loss ile gösterilebilir.",
      engineer: "Bu lab’da regresyon grafiği modelin öğrendiği eğriyi çizer.",
    },
    related: ["linear", "loss"],
  },
  {
    id: "classification",
    title: "Sınıflandırma",
    category: "task",
    summary: "Bir örneği sınıflardan birine atama görevidir.",
    example: "Çizimin 0 mı, 1 mi, gülen yüz mü olduğunu seçmek.",
    copy: {
      beginner: "Sınıflandırma, modelin seçenekler arasında karar vermesidir.",
      math: "Bu lab’da hedefler one-hot vektörle temsil edilir.",
      engineer: "En büyük output değeri tahmin edilen sınıf olarak okunur.",
    },
    related: ["sigmoid", "loss"],
  },
  {
    id: "normalization",
    title: "Normalizasyon",
    category: "data",
    summary: "Verileri ortak ölçeğe getirmektir.",
    formula: "x_norm = (x-min)/(max-min)",
    example: "Metrekare ve fiyat farklı ölçeklerdeyse eğitim zorlaşabilir.",
    copy: {
      beginner: "Normalizasyon, farklı büyüklükteki sayıları aynı oyun alanına koyar.",
      math: "Ölçek farkı gradient büyüklüklerini dengesiz yapabilir.",
      engineer: "Dataset Studio sayısal kolonları 0-1 aralığına taşıyabilir.",
    },
    related: ["gradient", "learning-rate"],
  },
];

export const ACTIVATION_CONCEPT_IDS: Record<ActivationName, ConceptId> = {
  input: "activation",
  linear: "linear",
  sigmoid: "sigmoid",
  tanh: "tanh",
  relu: "relu",
};

export function getConcept(id: ConceptId): Concept {
  return CONCEPTS.find((concept) => concept.id === id) ?? CONCEPTS[0];
}

export function conceptsByCategory(category: ConceptCategory): Concept[] {
  return CONCEPTS.filter((concept) => concept.category === category);
}
