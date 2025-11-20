package mathutils

import (
	"math"      // математические функции (exp и т.п.)
	"math/rand" // генератор случайных чисел
)

// Генерация матрицы случайных весов (распределение нормальное)
func RandomMatrix(rows, cols int) [][]float64 {
	scale := math.Sqrt(2.0 / float64(rows+cols)) // Xavier для сигмоиды

	m := make([][]float64, rows) // создаём массив строк
	for i := range m {
		m[i] = make([]float64, cols) // создаём столбцы
		for j := range m[i] {
			m[i][j] = rand.NormFloat64() * scale // маленькие случайные веса
		}
	}
	return m // возвращаем матрицу
}

// Матричное умножение A*B
func MatMul(a [][]float64, b [][]float64) [][]float64 {
	rowsA := len(a)    // число строк A
	colsA := len(a[0]) // число столбцов A
	colsB := len(b[0]) // число столбцов B

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
func AddBias(x [][]float64, b []float64) [][]float64 {
	out := make([][]float64, len(x)) // создаём новую матрицу
	for i := range x {
		out[i] = make([]float64, len(x[i]))
		for j := range x[i] {
			out[i][j] = x[i][j] + b[j] // прибавляем соответствующий bias
		}
	}
	return out
}
