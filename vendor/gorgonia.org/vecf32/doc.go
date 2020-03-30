// Package vecf32 provides common functions and methods for slices of float32.
//
// Name
//
// In the days of yore, scientists who computed with computers would use arrays to represent vectors, each value representing
// magnitude and/or direction. Then came the C++ Standard Templates Library, which sought to provide this data type in the standard
// library. Now, everyone conflates a term "vector" with dynamic arrays.
//
// In the C++ book, Bjarne Stroustrup has this to say:
// 		One could argue that valarray should have been called vector because is is a traditional mathematical vector
//		and that vector should have been called array.
//		However, this is not the way that the terminology evolved.
// 		A valarray is a vector optimized for numeric computation;
//		a vector is a flexible container designed for holding and manipulating objects of a wide variety of types;
//		and an array is a low-level built-in type
//
// Go has a better name for representing dynamically allocated arrays of any type - "slice". However, "slice" is both a noun and verb
// and many libraries that I use already use "slice"-as-a-verb as a name, so I had to settle for the second best name: "vector".
//
// It should be noted that while the names used in this package were definitely mathematically inspired, they bear only little resemblance
// the actual mathematical operations performed.
//
// Naming Convention
//
// The names of the operations assume you're working with slices of float32s. Hence `Add` performs elementwise addition between two []float32.
//
// Operations between []float32 and float32 are also supported, however they are differently named. Here are the equivalents:
/*
	+------------------------+--------------------------------------------+
	| []float32-[]float32 Op |         []float32-float32 Op               |
	+------------------------+--------------------------------------------+
	| Add(a, b []float32)    | Trans(a float32, b []float32)              |
	| Sub(a, b []float32)    | TransInv/TransInvR(a float32, b []float32) |
	| Mul(a, b []float32)    | Scale(a float32, b []float32)              |
	| Div(a, b []float32)    | ScaleInv/ScaleInvR(a float32, b []float32) |
	| Pow(a, b []float32)    | PowOf/PowOfR(a float32, b []float32)       |
	+------------------------+--------------------------------------------+
*/
// You may note that for the []float64 - float64 binary operations, the scalar (float64) is always the first operand. In operations
// that are not commutative, an additional function is provided, suffixed with "R" (for reverse)
//
// Range Check and BCE
//
// This package does not provide range checking. If indices are out of range, the functions will panic. This package should play well with BCE.
//
// TODO(anyone): provide SIMD vectorization for Incr and []float32-float64 functions
// Pull requests accepted
package vecf32 // import "gorgonia.org/vecf32"
