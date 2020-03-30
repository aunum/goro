// +build riscv

#include "textflag.h"

// func Exp(x float32) float32
TEXT ·Exp(SB),NOSPLIT,$0
	B ·exp(SB)


// func Log(x float64) float64
TEXT ·Log(SB),NOSPLIT,$0
	B ·log(SB)

TEXT ·Remainder(SB),NOSPLIT,$0
	B ·remainder(SB)

// func Sqrt(x float32) float32
TEXT ·Sqrt(SB),NOSPLIT,$0
	B ·sqrt(SB)
