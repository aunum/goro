// +build wasm

#include "textflag.h"

// func Exp(x float32) float32
TEXT ·Exp(SB),NOSPLIT,$0
	JMP ·exp(SB)


// func Log(x float64) float64
TEXT ·Log(SB),NOSPLIT,$0
	JMP ·log(SB)

TEXT ·Remainder(SB),NOSPLIT,$0
	JMP ·remainder(SB)

// func Sqrt(x float32) float32
TEXT ·Sqrt(SB),NOSPLIT,$0
	JMP ·sqrt(SB)
