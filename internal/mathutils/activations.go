package mathutils

import "math"

// Сигмоида — функция активации
func Sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x)) // формула σ(x) = 1/(1+e^-x)
}

// Производная сигмоиды для обратного прохода
func SigmoidDeriv(x float64) float64 {
	s := Sigmoid(x)    // сначала вычисляем сигмоиду
	return s * (1 - s) // её производная: s*(1-s)
}

// Сигмоида для матрицы
func SigmoidMatrix(m [][]float64) [][]float64 {
	out := make([][]float64, len(m))
	for i := range m {
		out[i] = make([]float64, len(m[i]))
		for j := range m[i] {
			out[i][j] = Sigmoid(m[i][j])
		}
	}
	return out
}
