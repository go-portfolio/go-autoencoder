package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/go-portfolio/go-neuro-autoencoder/internal/autoencoder"
)

func main() {
	rand.Seed(time.Now().UnixNano())

	inputSize := 8  // размер входных векторов
	latentSize := 8 // размер скрытого слоя
	epochs := 2000
	learningRate := 0.05

	ae := autoencoder.NewAutoencoder(inputSize, latentSize)

	// -----------------------------
	// Генерация обучающих данных
	// -----------------------------
	numTrain := 50
	trainBatch := generateBinaryBatch(numTrain, inputSize)

	// -----------------------------
	// Генерация тестовых данных
	// -----------------------------
	numTest := 5
	testBatch := generateBinaryBatch(numTest, inputSize)

	// -----------------------------
	// Обучение
	// -----------------------------
	for epoch := 0; epoch < epochs; epoch++ {
		loss := ae.TrainStep(trainBatch, learningRate)
		if epoch%200 == 0 {
			fmt.Printf("Epoch %d | Loss = %.6f\n", epoch, loss)
		}
	}

	// -----------------------------
	// Проверка на обучающих данных
	// -----------------------------
	fmt.Println("\n=== Проверка на обучающих данных ===")
	for i, x := range trainBatch {
		latent, out := encodeDecode(ae, x)
		fmt.Printf("Обучающие примеры %d:\n", i)
		fmt.Println("Входные данные:        ", x)
		fmt.Println("Внутреннее кодирование:", latent)
		fmt.Println("Восстановленное сетью:", out)
		fmt.Println()
	}

	// -----------------------------
	// Сохранение модели
	// -----------------------------
	err := ae.Save("autoencoder_weights.gob")
	if err != nil {
		fmt.Println("Error saving model:", err)
	} else {
		fmt.Println("Model saved to autoencoder_weights.gob")
	}

	// -----------------------------
	// Создаём новый автоэнкодер и загружаем веса
	// -----------------------------
	ae2 := autoencoder.NewAutoencoder(inputSize, latentSize)
	err = ae2.Load("autoencoder_weights.gob")
	if err != nil {
		fmt.Println("Error loading model:", err)
	} else {
		fmt.Println("Model loaded successfully")
	}

	// -----------------------------
	// Проверка на тестовых данных
	// -----------------------------
	fmt.Println("\n=== Проверка на тестовых данных ===")
	for i, x := range testBatch {
		latent, out := encodeDecode(ae2, x)
		fmt.Printf("Тестовые примеры %d:\n", i)
		fmt.Println("Входные данные:        ", x)
		fmt.Println("Внутреннее кодирование:", latent)
		fmt.Println("Восстановленное сетью:", out)
		fmt.Println()
	}
}

// -------------------------------------
// Вспомогательные функции
// -------------------------------------

// generateBinaryBatch — создаёт случайные бинарные вектора
func generateBinaryBatch(n, size int) [][]float64 {
	batch := make([][]float64, n)
	for i := 0; i < n; i++ {
		sample := make([]float64, size)
		for j := 0; j < size; j++ {
			sample[j] = float64(rand.Intn(2))
		}
		batch[i] = sample
	}
	return batch
}

// encodeDecode — чистая кодировка → декодировка
func encodeDecode(ae *autoencoder.Autoencoder, x []float64) ([]float64, []float64) {
	latent, out, _, _ := ae.Forward([][]float64{x})
	return latent[0], binarize(out[0])
}

// binarize — округляет выход до 0/1
func binarize(vec []float64) []float64 {
	res := make([]float64, len(vec))
	for i := range vec {
		if vec[i] > 0.5 {
			res[i] = 1
		} else {
			res[i] = 0
		}
	}
	return res
}
