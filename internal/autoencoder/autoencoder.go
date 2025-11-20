package autoencoder

import "github.com/go-portfolio/go-neuro-autoencoder/internal/mathutils"

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
// Возвращает:
// - a1: активации скрытого слоя
// - out: выход автоэнкодера
// - z1: линейное преобразование входа (xW1 + b1) до активации
// - z2: линейное преобразование скрытого слоя (a1W2 + b2) до активации
func (ae *Autoencoder) Forward(x [][]float64) ([][]float64, [][]float64, [][]float64, [][]float64) {
	z1 := mathutils.AddBias(mathutils.MatMul(x, ae.W1), ae.b1)

	a1 := make([][]float64, len(z1))
	for i := range z1 {
		a1[i] = make([]float64, len(z1[i]))
		for j := range z1[i] {
			a1[i][j] = mathutils.Sigmoid(z1[i][j])
		}
	}

	z2 := mathutils.AddBias(mathutils.MatMul(a1, ae.W2), ae.b2)

	out := make([][]float64, len(z2))
	for i := range z2 {
		out[i] = make([]float64, len(z2[i]))
		for j := range z2[i] {
			out[i][j] = mathutils.Sigmoid(z2[i][j])
		}
	}

	// Применяем порог 0.5 для бинаризации выхода
	for i := range out[0] {
		if out[0][i] > 0.5 {
			out[0][i] = 1
		} else {
			out[0][i] = 0
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
		for j := 0; j < ae.latentSize; j++ {
			sum := 0.0
			for k := 0; k < ae.inputSize; k++ {
				sum += dOut[i][k] * ae.W2[j][k]
			}
			dA1[i][j] = sum * mathutils.SigmoidDeriv(z1[i][j])
		}
	}
	return dA1
}

// computeGradientsW1 вычисляет градиенты весов и bias энкодера.
func (ae *Autoencoder) computeGradientsW1(x, dA1 [][]float64) ([][]float64, []float64) {
	dW1 := make([][]float64, ae.inputSize)
	for i := range dW1 {
		dW1[i] = make([]float64, ae.latentSize)
	}
	db1 := make([]float64, ae.latentSize)

	for i := 0; i < len(x); i++ {
		for j := 0; j < ae.latentSize; j++ {
			db1[j] += dA1[i][j]
			for k := 0; k < ae.inputSize; k++ {
				dW1[k][j] += x[i][k] * dA1[i][j]
			}
		}
	}
	return dW1, db1
}

// updateWeights обновляет веса и bias автоэнкодера с использованием градиентов и скорости обучения lr.
func (ae *Autoencoder) updateWeights(dW1 [][]float64, db1 []float64, dW2 [][]float64, db2 []float64, lr float64) {
	for i := 0; i < ae.inputSize; i++ {
		for j := 0; j < ae.latentSize; j++ {
			ae.W1[i][j] -= lr * dW1[i][j]
		}
	}
	for j := 0; j < ae.latentSize; j++ {
		ae.b1[j] -= lr * db1[j]
	}

	for i := 0; i < ae.latentSize; i++ {
		for j := 0; j < ae.inputSize; j++ {
			ae.W2[i][j] -= lr * dW2[i][j]
		}
	}
	for j := 0; j < ae.inputSize; j++ {
		ae.b2[j] -= lr * db2[j]
	}
}
