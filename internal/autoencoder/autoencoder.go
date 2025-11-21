package autoencoder

import (
	"encoding/gob"
	"os"
	"runtime"
	"sync"

	"github.com/go-portfolio/go-neuro-autoencoder/internal/mathutils"
)

// Autoencoder представляет простой полносвязный автоэнкодер с одним скрытым слоем.
type Autoencoder struct {
	inputSize  int // размер входного вектора
	latentSize int // размер скрытого (латентного) слоя

	W1 [][]float64 // веса энкодера (вход → скрытый)
	b1 []float64   // bias скрытого слоя

	W2 [][]float64 // веса декодера (скрытый → выход)
	b2 []float64   // bias выходного слоя
}

// Encode кодирует входной вектор x в скрытое представление.
func (ae *Autoencoder) Encode(x [][]float64) [][]float64 {
	z1 := mathutils.MatMul(x, ae.W1)  // линейная трансформация
	a1 := mathutils.SigmoidMatrix(z1) // активация
	return a1
}

// Decode восстанавливает данные из скрытого представления.
func (ae *Autoencoder) Decode(latent [][]float64) [][]float64 {
	z2 := mathutils.MatMul(latent, ae.W2)
	out := mathutils.SigmoidMatrix(z2)
	return out
}

// NewAutoencoder создаёт новый автоэнкодер с заданными размерами входного и скрытого слоя.
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
	z1 := mathutils.AddBias(mathutils.MatMul(x, ae.W1), ae.b1)
	a1 := mathutils.SigmoidMatrix(z1)

	// --- Декодер ---
	z2 := mathutils.AddBias(mathutils.MatMul(a1, ae.W2), ae.b2)
	out := mathutils.SigmoidMatrix(z2)

	return a1, out, z1, z2
}

// TrainStep выполняет один шаг обучения с использованием батча данных x.
func (ae *Autoencoder) TrainStep(x [][]float64, lr float64) float64 {
	a1, out, z1, z2 := ae.Forward(x)

	// Параллельное вычисление градиентов
	dOut, mse := ae.computeOutputGradient(out, x, z2)
	dW2, db2 := ae.computeGradientsW2(a1, dOut)
	dA1 := ae.backpropHidden(dOut, z1)
	dW1, db1 := ae.computeGradientsW1(x, dA1)

	// Параллельное обновление весов
	ae.updateWeights(dW1, db1, dW2, db2, lr)

	return mse
}

// computeOutputGradient вычисляет градиент ошибки на выходе и MSE.
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

// computeGradientsW2 вычисляет градиенты для весов W2 и bias b2.
func (ae *Autoencoder) computeGradientsW2(a1, dOut [][]float64) ([][]float64, []float64) {
	dW2 := make([][]float64, ae.latentSize)
	for i := range dW2 {
		dW2[i] = make([]float64, ae.inputSize)
	}
	db2 := make([]float64, ae.inputSize)

	var wg sync.WaitGroup
	numWorkers := runtime.NumCPU()

	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			start := w * len(a1) / numWorkers
			end := (w + 1) * len(a1) / numWorkers
			for i := start; i < end; i++ {
				for j := 0; j < ae.inputSize; j++ {
					db2[j] += dOut[i][j]
					for k := 0; k < ae.latentSize; k++ {
						dW2[k][j] += a1[i][k] * dOut[i][j]
					}
				}
			}
		}(w)
	}
	wg.Wait()
	return dW2, db2
}

// backpropHidden выполняет обратное распространение ошибки через скрытый слой.
func (ae *Autoencoder) backpropHidden(dOut, z1 [][]float64) [][]float64 {
	dA1 := make([][]float64, len(dOut))
	for i := range dA1 {
		dA1[i] = make([]float64, ae.latentSize)
	}

	var wg sync.WaitGroup
	numWorkers := runtime.NumCPU()

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
	wg.Wait()
	return dA1
}

// computeGradientsW1 вычисляет градиенты для весов W1 и bias b1.
func (ae *Autoencoder) computeGradientsW1(x, dA1 [][]float64) ([][]float64, []float64) {
	dW1 := make([][]float64, ae.inputSize)
	for i := range dW1 {
		dW1[i] = make([]float64, ae.latentSize)
	}
	db1 := make([]float64, ae.latentSize)

	var wg sync.WaitGroup
	numWorkers := runtime.NumCPU()

	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			start := w * len(x) / numWorkers
			end := (w + 1) * len(x) / numWorkers
			for i := start; i < end; i++ {
				for j := 0; j < ae.latentSize; j++ {
					db1[j] += dA1[i][j]
					for k := 0; k < ae.inputSize; k++ {
						dW1[k][j] += x[i][k] * dA1[i][j]
					}
				}
			}
		}(w)
	}
	wg.Wait()
	return dW1, db1
}

// updateWeights обновляет веса и bias автоэнкодера с использованием градиентов.
func (ae *Autoencoder) updateWeights(dW1 [][]float64, db1 []float64, dW2 [][]float64, db2 []float64, lr float64) {
	var wg sync.WaitGroup

	for i := 0; i < ae.inputSize; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			for j := 0; j < ae.latentSize; j++ {
				ae.W1[i][j] -= lr * dW1[i][j]
			}
		}(i)
	}

	wg.Add(1)
	go func() {
		defer wg.Done()
		for j := 0; j < ae.latentSize; j++ {
			ae.b1[j] -= lr * db1[j]
		}
	}()

	for i := 0; i < ae.latentSize; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			for j := 0; j < ae.inputSize; j++ {
				ae.W2[i][j] -= lr * dW2[i][j]
			}
		}(i)
	}

	wg.Add(1)
	go func() {
		defer wg.Done()
		for j := 0; j < ae.inputSize; j++ {
			ae.b2[j] -= lr * db2[j]
		}
	}()

	wg.Wait()
}

// Save сохраняет веса и bias автоэнкодера в файл.
func (ae *Autoencoder) Save(filename string) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	encoder := gob.NewEncoder(file)
	return encoder.Encode(ae)
}

// Load загружает веса и bias из файла.
func (ae *Autoencoder) Load(filename string) error {
	file, err := os.Open(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	decoder := gob.NewDecoder(file)
	return decoder.Decode(ae)
}
