package autoencoder

import (
	"encoding/gob"
	"os"
	"runtime"
	"sync"

	"github.com/go-portfolio/go-neuro-autoencoder/internal/mathutils"
)

// Autoencoder представляет простой полносвязный автоэнкодер с одним скрытым слоем.
// Состоит из энкодера (W1, b1) и декодера (W2, b2).
type Autoencoder struct {
	inputSize  int // размер входного вектора
	latentSize int // размер скрытого (латентного) слоя

	W1 [][]float64 // веса энкодера (вход → скрытый)
	b1 []float64   // bias скрытого слоя

	W2 [][]float64 // веса декодера (скрытый → выход)
	b2 []float64   // bias выходного слоя
}

func (ae *Autoencoder) Encode(x [][]float64) [][]float64 {
	z1 := mathutils.MatMul(x, ae.W1)  // линейная трансформация
	a1 := mathutils.SigmoidMatrix(z1) // активация
	return a1
}

func (ae *Autoencoder) Decode(latent [][]float64) [][]float64 {
	z2 := mathutils.MatMul(latent, ae.W2)
	out := mathutils.SigmoidMatrix(z2)
	return out
}

// NewAutoencoder создаёт новый автоэнкодер с заданными размерами входного и скрытого слоя.
// Веса инициализируются случайно с помощью Xavier/He, bias инициализируются нулями.
func NewAutoencoder(inputSize, latentSize int) *Autoencoder {
	return &Autoencoder{
		inputSize:  inputSize,
		latentSize: latentSize,

		W1: mathutils.RandomMatrix(inputSize, latentSize),
		b1: make([]float64, latentSize),

		W2: mathutils.RandomMatrix(latentSize, inputSize),
		b2: make([]float64, inputSize),
	}
}

// Forward выполняет прямой проход автоэнкодера для батча входных данных x.
func (ae *Autoencoder) Forward(x [][]float64) ([][]float64, [][]float64, [][]float64, [][]float64) {

	// --- Энкодер ---
	// z1 = xW1 + b1
	z1 := mathutils.AddBias(mathutils.MatMul(x, ae.W1), ae.b1)

	// a1 = sigmoid(z1)
	a1 := make([][]float64, len(z1))
	for i := range z1 {
		a1[i] = make([]float64, len(z1[i]))
		for j := range z1[i] {
			a1[i][j] = mathutils.Sigmoid(z1[i][j])
		}
	}

	// --- Декодер ---
	// z2 = a1W2 + b2
	z2 := mathutils.AddBias(mathutils.MatMul(a1, ae.W2), ae.b2)

	// out = sigmoid(z2)
	out := make([][]float64, len(z2))
	for i := range z2 {
		out[i] = make([]float64, len(z2[i]))
		for j := range z2[i] {
			out[i][j] = mathutils.Sigmoid(z2[i][j])
		}
	}

	return a1, out, z1, z2
}

// TrainStep выполняет один шаг обучения автоэнкодера на батче x с заданной скоростью обучения lr.
// Выполняет forward pass, вычисление градиентов методом обратного распространения и обновление весов.
// Возвращает среднеквадратичную ошибку (MSE) по батчу.
func (ae *Autoencoder) TrainStep(x [][]float64, lr float64) float64 {
	a1, out, z1, z2 := ae.Forward(x)

	dOut, mse := ae.computeOutputGradient(out, x, z2)
	dW2, db2 := ae.computeGradientsW2(a1, dOut)
	dA1 := ae.backpropHidden(dOut, z1)
	dW1, db1 := ae.computeGradientsW1(x, dA1)

	ae.updateWeights(dW1, db1, dW2, db2, lr)

	return mse
}

// computeOutputGradient вычисляет градиент ошибки на выходе и среднеквадратичную ошибку (MSE).
func (ae *Autoencoder) computeOutputGradient(out, x, z2 [][]float64) ([][]float64, float64) {
	mse := 0.0
	dOut := make([][]float64, len(out))
	for i := range out {
		dOut[i] = make([]float64, len(out[i]))
		for j := range out[i] {
			diff := out[i][j] - x[i][j]
			mse += diff * diff
			dOut[i][j] = 2 * diff * mathutils.SigmoidDeriv(z2[i][j])
		}
	}
	mse /= float64(len(out[0]))
	return dOut, mse
}

// computeGradientsW2 вычисляет градиенты весов и bias декодера.
func (ae *Autoencoder) computeGradientsW2(a1, dOut [][]float64) ([][]float64, []float64) {
	dW2 := make([][]float64, ae.latentSize)
	for i := range dW2 {
		dW2[i] = make([]float64, ae.inputSize)
	}
	db2 := make([]float64, ae.inputSize)

	for i := 0; i < len(a1); i++ {
		for j := 0; j < ae.inputSize; j++ {
			db2[j] += dOut[i][j]
			for k := 0; k < ae.latentSize; k++ {
				dW2[k][j] += a1[i][k] * dOut[i][j]
			}
		}
	}
	return dW2, db2
}

// backpropHidden выполняет обратное распространение ошибки через скрытый слой.
func (ae *Autoencoder) backpropHidden(dOut, z1 [][]float64) [][]float64 {
	dA1 := make([][]float64, len(dOut))
	for i := range dA1 {
		dA1[i] = make([]float64, ae.latentSize)
	}

	var wg sync.WaitGroup
	numWorkers := runtime.NumCPU() // или len(dOut) если батч меньше

	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			for j := 0; j < ae.latentSize; j++ {
				sum := 0.0
				for k := 0; k < ae.inputSize; k++ {
					sum += dOut[i][k] * ae.W2[j][k]
				}
				dA1[i][j] = sum * mathutils.SigmoidDeriv(z1[i][j])
			}
		}(i)
	}

	wg.Wait() // ждём, пока все горутины завершатся
	return dA1
}

// Параллельные вычисления выполняются по батчу и по нейронам.
func (ae *Autoencoder) computeGradientsW1(x, dA1 [][]float64) ([][]float64, []float64) {
	dW1 := make([][]float64, ae.inputSize)
	for i := range dW1 {
		dW1[i] = make([]float64, ae.latentSize)
	}
	db1 := make([]float64, ae.latentSize)

	numWorkers := runtime.NumCPU()
	wg := sync.WaitGroup{}
	chunkSize := (len(x) + numWorkers - 1) / numWorkers

	for w := 0; w < numWorkers; w++ {
		start := w * chunkSize
		end := start + chunkSize
		if end > len(x) {
			end = len(x)
		}

		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			localDW := make([][]float64, ae.inputSize)
			for i := range localDW {
				localDW[i] = make([]float64, ae.latentSize)
			}
			localDB := make([]float64, ae.latentSize)

			for i := start; i < end; i++ {
				for j := 0; j < ae.latentSize; j++ {
					localDB[j] += dA1[i][j]
					for k := 0; k < ae.inputSize; k++ {
						localDW[k][j] += x[i][k] * dA1[i][j]
					}
				}
			}

			// Суммируем локальные градиенты в глобальные
			for i := 0; i < ae.inputSize; i++ {
				for j := 0; j < ae.latentSize; j++ {
					dW1[i][j] += localDW[i][j]
				}
			}
			for j := 0; j < ae.latentSize; j++ {
				db1[j] += localDB[j]
			}
		}(start, end)
	}

	wg.Wait()
	return dW1, db1
}

// updateWeights обновляет веса и bias автоэнкодера с использованием градиентов и скорости обучения lr.

func (ae *Autoencoder) updateWeights(dW1 [][]float64, db1 []float64, dW2 [][]float64, db2 []float64, lr float64) {
	var wg sync.WaitGroup

	// Обновляем W1 параллельно
	for i := 0; i < ae.inputSize; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			for j := 0; j < ae.latentSize; j++ {
				ae.W1[i][j] -= lr * dW1[i][j]
			}
		}(i)
	}

	// Обновляем b1 параллельно
	wg.Add(1)
	go func() {
		defer wg.Done()
		for j := 0; j < ae.latentSize; j++ {
			ae.b1[j] -= lr * db1[j]
		}
	}()

	// Обновляем W2 параллельно
	for i := 0; i < ae.latentSize; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			for j := 0; j < ae.inputSize; j++ {
				ae.W2[i][j] -= lr * dW2[i][j]
			}
		}(i)
	}

	// Обновляем b2 параллельно
	wg.Add(1)
	go func() {
		defer wg.Done()
		for j := 0; j < ae.inputSize; j++ {
			ae.b2[j] -= lr * db2[j]
		}
	}()

	// Ждём завершения всех горутин
	wg.Wait()
}

// Save сохраняет веса и bias автоэнкодера в файл
func (ae *Autoencoder) Save(filename string) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	encoder := gob.NewEncoder(file)
	return encoder.Encode(ae)
}

// Load загружает веса и bias из файла
func (ae *Autoencoder) Load(filename string) error {
	file, err := os.Open(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	decoder := gob.NewDecoder(file)
	return decoder.Decode(ae)
}
