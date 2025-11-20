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
