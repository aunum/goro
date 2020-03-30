// Package require provides methods for enforcing requirements on values or exiting.
package require

import (
	"github.com/aunum/log"
)

// Nil requires that the given value is nil or exists.
func Nil(v interface{}) {
	if v != nil {
		log.Fatalf("%v must be nil", v)
	}
}

// NoError requires that the error is nil or exists.
func NoError(err error) {
	if err != nil {
		log.Fatal(err)
	}
}
