package main // объявляем основной пакет, точка входа программы

import (
    "fmt"       // пакет для вывода на экран
    "math/rand" // генератор случайных чисел
    "math"      // математические функции (exp и т.п.)
)

// -------------------- Utility functions --------------------

// Сигмоида — функция активации
func sigmoid(x float64) float64 {
    return 1.0 / (1.0 + math.Exp(-x)) // формула σ(x) = 1/(1+e^-x)
}

// Производная сигмоиды для обратного прохода
func sigmoidDeriv(x float64) float64 {
    s := sigmoid(x)     // сначала вычисляем сигмоиду
    return s * (1 - s)  // её производная: s*(1-s)
}

// Генерация матрицы случайных весов (распределение нормальное)
func randomMatrix(rows, cols int) [][]float64 {
    m := make([][]float64, rows) // создаём массив строк
    for i := range m {
        m[i] = make([]float64, cols) // создаём столбцы
        for j := range m[i] {
            m[i][j] = rand.NormFloat64() * 0.1 // маленькие случайные веса
        }
    }
    return m // возвращаем матрицу
}

// Матричное умножение A*B
func matMul(a [][]float64, b [][]float64) [][]float64 {
    rowsA := len(a)       // число строк A
    colsA := len(a[0])    // число столбцов A
    colsB := len(b[0])    // число столбцов B

    // создаём матрицу результата
    out := make([][]float64, rowsA)
    for i := range out {
        out[i] = make([]float64, colsB)
    }

    // тройной цикл умножения матриц
    for i := 0; i < rowsA; i++ {
        for j := 0; j < colsB; j++ {
            sum := 0.0
            for k := 0; k < colsA; k++ {
                sum += a[i][k] * b[k][j] // умножение и суммирование
            }
            out[i][j] = sum // записываем результат
        }
    }
    return out
}

// Добавление смещения (bias) к каждому ряду матрицы
func addBias(x [][]float64, b []float64) [][]float64 {
    out := make([][]float64, len(x)) // создаём новую матрицу
    for i := range x {
        out[i] = make([]float64, len(x[i]))
        for j := range x[i] {
            out[i][j] = x[i][j] + b[j] // прибавляем соответствующий bias
        }
    }
    return out
}

// -------------------- Autoencoder struct --------------------

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

        W1: randomMatrix(inputSize, latentSize), // создаём веса энкодера
        b1: make([]float64, latentSize),         // bias инициализируется нулями

        W2: randomMatrix(latentSize, inputSize), // создаём веса декодера
        b2: make([]float64, inputSize),          // bias декодера — нули
    }
}

// Прямой проход: возвращает скрытый слой, выход и z1/z2 до сигмоиды
func (ae *Autoencoder) Forward(x [][]float64) ([][]float64, [][]float64, [][]float64, [][]float64) {

    // z1 = xW1 + b1 (линейный слой энкодера)
    z1 := addBias(matMul(x, ae.W1), ae.b1)

    // a1 = sigmoid(z1) (активация скрытого слоя)
    a1 := make([][]float64, len(z1))
    for i := range z1 {
        a1[i] = make([]float64, len(z1[i]))
        for j := range z1[i] {
            a1[i][j] = sigmoid(z1[i][j])
        }
    }

    // z2 = a1W2 + b2 (линейный слой декодера)
    z2 := addBias(matMul(a1, ae.W2), ae.b2)

    // out = sigmoid(z2) — финальный выход
    out := make([][]float64, len(z2))
    for i := range z2 {
        out[i] = make([]float64, len(z2[i]))
        for j := range z2[i] {
            out[i][j] = sigmoid(z2[i][j])
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
            diff := out[i][j] - x[i][j] // ошибка реконструкции
            mse += diff * diff          // накапливаем MSE
            dOut[i][j] = 2 * diff * sigmoidDeriv(z2[i][j]) // градиент выхода
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
            dA1[i][j] = sum * sigmoidDeriv(z1[i][j]) // учитываем производную сигмоиды
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

// -------------------- Main training loop --------------------

func main() {
    inputSize := 8   // размер входного вектора
    latentSize := 3  // хотим сжать до 3 значений
    ae := NewAutoencoder(inputSize, latentSize) // создаём автоэнкодер

    // Один тренировочный пример — бинарный вектор
    x := []float64{0, 1, 0, 1, 0, 1, 0, 1}
    batch := [][]float64{x} // оформляем как батч (1 пример)

    // Обучаем 2000 эпох
    for epoch := 0; epoch < 2000; epoch++ {
        loss := ae.TrainStep(batch, 0.05) // шаг оптимизации

        if epoch%200 == 0 {               // каждые 200 эпох выводим ошибку
            fmt.Println("Epoch", epoch, "Loss:", loss)
        }
    }

    // Проверяем результат обучения
    _, out, _, _ := ae.Forward(batch)

    fmt.Println("Input:      ", x)        // печатаем вход
    fmt.Println("Reconstructed:", out[0]) // печатаем восстановленный вектор
}
