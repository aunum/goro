package hm

import "fmt"

// A Constraint is well.. a constraint that says a must equal to b. It's used mainly in the constraint generation process.
type Constraint struct {
	a, b Type
}

func (c Constraint) Apply(sub Subs) Substitutable {
	c.a = c.a.Apply(sub).(Type)
	c.b = c.b.Apply(sub).(Type)
	return c
}

func (c Constraint) FreeTypeVar() TypeVarSet {
	var retVal TypeVarSet
	retVal = c.a.FreeTypeVar().Union(retVal)
	retVal = c.b.FreeTypeVar().Union(retVal)
	return retVal
}

func (c Constraint) Format(state fmt.State, r rune) {
	fmt.Fprintf(state, "{%v = %v}", c.a, c.b)
}
