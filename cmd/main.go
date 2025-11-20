package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/go-portfolio/go-neuro-autoencoder/internal/autoencoder"
)

func main() {
	rand.Seed(time.Now().UnixNano())

	inputSize := 8
	latentSize := 3
	ae := autoencoder.NewAutoencoder(inputSize, latentSize)

	// --- Генерация обучающих данных ---
	numTrain := 10
	trainBatch := make([][]float64, numTrain)
	for i := 0; i < numTrain; i++ {
		sample := make([]float64, inputSize)
		for j := 0; j < inputSize; j++ {
			sample[j] = float64(rand.Intn(2))
		}
		trainBatch[i] = sample
	}

	// --- Генерация тестовых данных (не видимых при обучении) ---
	numTest := 5
	testBatch := make([][]float64, numTest)
	for i := 0; i < numTest; i++ {
		sample := make([]float64, inputSize)
		for j := 0; j < inputSize; j++ {
			sample[j] = float64(rand.Intn(2))
		}
		testBatch[i] = sample
	}

	// --- Обучение автоэнкодера ---
	epochs := 2000
	learningRate := 0.05

	for epoch := 0; epoch < epochs; epoch++ {
		loss := ae.TrainStep(trainBatch, learningRate)
		if epoch%200 == 0 {
			fmt.Printf("Epoch %d Loss: %.6f\n", epoch, loss)
		}
	}

	// --- Проверка на обучающих данных ---
	fmt.Println("\n=== Reconstruction on training data ===")
	for i, x := range trainBatch {
		_, out, _, _ := ae.Forward([][]float64{x})
		fmt.Printf("Train Sample %d:\n", i)
		fmt.Println("Input:        ", x)
		fmt.Println("Reconstructed:", out[0])
		fmt.Println()
	}

	// --- Проверка на тестовых данных ---
	fmt.Println("\n=== Reconstruction on test data ===")
	for i, x := range testBatch {
		_, out, _, _ := ae.Forward([][]float64{x})
		fmt.Printf("Test Sample %d:\n", i)
		fmt.Println("Input:        ", x)
		fmt.Println("Reconstructed:", out[0])
		fmt.Println()
	}
}
