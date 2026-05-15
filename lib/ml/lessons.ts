import type { LessonStep } from "./lab-types";
import type { Task, TaskId } from "./tasks";

const baseLessons: LessonStep[] = [
  {
    id: "data",
    title: "Veriyi oku",
    text: "Her satır bir örnek. Giriş değerleri input nöronlarına, hedef değerleri loss hesabına gider.",
  },
  {
    id: "forward",
    title: "Forward hesabını izle",
    text: "Bir step seç ve nöronun Σ(x*w)+b hesabını yakından incele.",
  },
  {
    id: "loss",
    title: "Loss neden oluştu?",
    text: "Tahmin ile hedef arasındaki fark loss değerini üretir. Loss küçüldükçe model daha iyi uyum sağlar.",
  },
  {
    id: "backprop",
    title: "Hata geriye nasıl dağılıyor?",
    text: "Backprop step’lerinde delta değerlerinin önce output, sonra hidden nöronlara aktığını izle.",
  },
  {
    id: "update",
    title: "Ağırlık güncellemesini çöz",
    text: "Gradient, learning rate ile çarpılır ve ağırlık ters yönde hareket eder.",
  },
];

const imageLessons: LessonStep[] = [
  {
    id: "pixels",
    title: "Pikseli input olarak düşün",
    text: "5x5 karedeki her piksel bir input nöronudur. Ağ, piksel deseninden sınıf sinyali üretir.",
  },
  ...baseLessons,
];

export function lessonsForTask(task: Task): LessonStep[] {
  const lessonsByTask: Partial<Record<TaskId, LessonStep[]>> = {
    digit: imageLessons,
  };
  return lessonsByTask[task.id] ?? baseLessons;
}
