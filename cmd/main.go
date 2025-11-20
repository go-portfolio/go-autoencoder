package main // объявляем основной пакет, точка входа программы

import (
	"fmt" // пакет для вывода на экран
	// математические функции (exp и т.п.)
	// генератор случайных чисел
	"github.com/go-portfolio/go-neuro-autoencoder/internal/autoencoder"
)

// -------------------- Utility functions --------------------

// -------------------- Main training loop --------------------

func main() {
	inputSize := 8                                          // размер входного вектора
	latentSize := 3                                         // хотим сжать до 3 значений
	ae := autoencoder.NewAutoencoder(inputSize, latentSize) // создаём автоэнкодер

	// Один тренировочный пример — бинарный вектор
	x := []float64{0, 1, 0, 1, 0, 1, 0, 1}
	batch := [][]float64{x} // оформляем как батч (1 пример)

	// Обучаем 2000 эпох
	for epoch := 0; epoch < 2000; epoch++ {
		loss := ae.TrainStep(batch, 0.05) // шаг оптимизации

		if epoch%200 == 0 { // каждые 200 эпох выводим ошибку
			fmt.Println("Epoch", epoch, "Loss:", loss)
		}
	}

	// Проверяем результат обучения
	_, out, _, _ := ae.Forward(batch)

	fmt.Println("Input:      ", x)        // печатаем вход
	fmt.Println("Reconstructed:", out[0]) // печатаем восстановленный вектор
}
