package autoencoder

import "github.com/go-portfolio/go-neuro-autoencoder/internal/mathutils"

// Структура автоэнкодера — хранит размеры и веса
type Autoencoder struct {
	inputSize  int // размер входного вектора
	latentSize int // размер сжатого представления

	// Параметры энкодера
	W1 [][]float64 // веса вход → латент
	b1 []float64   // bias для скрытого слоя

	// Параметры декодера
	W2 [][]float64 // веса латент → выход
	b2 []float64   // bias выходного слоя
}

// Создание нового автоэнкодера со случайными весами
func NewAutoencoder(inputSize, latentSize int) *Autoencoder {
	return &Autoencoder{
		inputSize:  inputSize,  // записываем размер входа
		latentSize: latentSize, // размер скрытого слоя

		W1: mathutils.RandomMatrix(inputSize, latentSize), // создаём веса энкодера
		b1: make([]float64, latentSize),                   // bias инициализируется нулями

		W2: mathutils.RandomMatrix(latentSize, inputSize), // создаём веса декодера
		b2: make([]float64, inputSize),                    // bias декодера — нули
	}
}

// Прямой проход: возвращает скрытый слой, выход и z1/z2 до сигмоиды
func (ae *Autoencoder) Forward(x [][]float64) ([][]float64, [][]float64, [][]float64, [][]float64) {

	// z1 = xW1 + b1 (линейный слой энкодера)
	z1 := mathutils.AddBias(mathutils.MatMul(x, ae.W1), ae.b1)

	// a1 = sigmoid(z1) (активация скрытого слоя)
	a1 := make([][]float64, len(z1))
	for i := range z1 {
		a1[i] = make([]float64, len(z1[i]))
		for j := range z1[i] {
			a1[i][j] = mathutils.Sigmoid(z1[i][j])
		}
	}

	// z2 = a1W2 + b2 (линейный слой декодера)
	z2 := mathutils.AddBias(mathutils.MatMul(a1, ae.W2), ae.b2)

	// out = sigmoid(z2) — финальный выход
	out := make([][]float64, len(z2))
	for i := range z2 {
		out[i] = make([]float64, len(z2[i]))
		for j := range z2[i] {
			out[i][j] = mathutils.Sigmoid(z2[i][j])
		}
	}

	return a1, out, z1, z2 // возвращаем всё для backprop
}

// Один шаг тренировки (SGD со своим backprop)
func (ae *Autoencoder) TrainStep(x [][]float64, lr float64) float64 {
	// Выполняем forward
	a1, out, z1, z2 := ae.Forward(x)

	// Вычисляем MSE и градиент dOut
	mse := 0.0
	dOut := make([][]float64, len(out))
	for i := range out {
		dOut[i] = make([]float64, len(out[i]))
		for j := range out[i] {
			diff := out[i][j] - x[i][j]                              // ошибка реконструкции
			mse += diff * diff                                       // накапливаем MSE
			dOut[i][j] = 2 * diff * mathutils.SigmoidDeriv(z2[i][j]) // градиент выхода
		}
	}
	mse /= float64(len(out[0])) // нормируем ошибку

	// --- Градиенты для второго слоя (W2, b2) ---
	dW2 := make([][]float64, ae.latentSize)
	for i := range dW2 {
		dW2[i] = make([]float64, ae.inputSize)
	}
	db2 := make([]float64, ae.inputSize)

	// Вычисляем производные по W2 и b2
	for i := 0; i < len(a1); i++ {
		for j := 0; j < ae.inputSize; j++ {
			db2[j] += dOut[i][j] // градиент bias
			for k := 0; k < ae.latentSize; k++ {
				dW2[k][j] += a1[i][k] * dOut[i][j] // градиент веса
			}
		}
	}

	// --- Backprop в скрытый слой ---
	dA1 := make([][]float64, len(a1))
	for i := range dA1 {
		dA1[i] = make([]float64, ae.latentSize)
		for j := 0; j < ae.latentSize; j++ {
			sum := 0.0
			for k := 0; k < ae.inputSize; k++ {
				sum += dOut[i][k] * ae.W2[j][k] // dL/d(a1)
			}
			dA1[i][j] = sum * mathutils.SigmoidDeriv(z1[i][j]) // учитываем производную сигмоиды
		}
	}

	// --- Градиенты первого слоя (W1, b1) ---
	dW1 := make([][]float64, ae.inputSize)
	for i := range dW1 {
		dW1[i] = make([]float64, ae.latentSize)
	}
	db1 := make([]float64, ae.latentSize)

	// Вычисляем градиенты по W1 и b1
	for i := 0; i < len(x); i++ {
		for j := 0; j < ae.latentSize; j++ {
			db1[j] += dA1[i][j] // градиент bias
			for k := 0; k < ae.inputSize; k++ {
				dW1[k][j] += x[i][k] * dA1[i][j] // градиент веса
			}
		}
	}

	// --- Обновление весов SGD ---
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

	return mse // возвращаем ошибку
}
