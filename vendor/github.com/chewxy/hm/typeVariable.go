package hm

import (
	"fmt"

	"github.com/pkg/errors"
)

// TypeVariable is a variable that ranges over the types - that is to say it can take any type.
type TypeVariable rune

func (t TypeVariable) Name() string { return string(t) }
func (t TypeVariable) Apply(sub Subs) Substitutable {
	if sub == nil {
		return t
	}

	if retVal, ok := sub.Get(t); ok {
		return retVal
	}

	return t
}

func (t TypeVariable) FreeTypeVar() TypeVarSet { tvs := BorrowTypeVarSet(1); tvs[0] = t; return tvs }
func (t TypeVariable) Normalize(k, v TypeVarSet) (Type, error) {
	if i := k.Index(t); i >= 0 {
		return v[i], nil
	}
	return nil, errors.Errorf("Type Variable %v not in signature", t)
}

func (t TypeVariable) Types() Types               { return nil }
func (t TypeVariable) String() string             { return string(t) }
func (t TypeVariable) Format(s fmt.State, c rune) { fmt.Fprintf(s, "%c", rune(t)) }
func (t TypeVariable) Eq(other Type) bool         { return other == t }
