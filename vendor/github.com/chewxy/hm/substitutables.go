package hm

import "fmt"

// Constraints is a slice of Constraint. Like a Constraint, it is also a Substitutable
type Constraints []Constraint

func (cs Constraints) Apply(sub Subs) Substitutable {
	// an optimization
	if sub == nil {
		return cs
	}

	if len(cs) == 0 {
		return cs
	}

	logf("Constraints: %d", len(cs))
	logf("Applying %v to %v", sub, cs)
	for i, c := range cs {
		cs[i] = c.Apply(sub).(Constraint)
	}
	logf("Constraints %v", cs)
	return cs
}

func (cs Constraints) FreeTypeVar() TypeVarSet {
	var retVal TypeVarSet
	for _, v := range cs {
		retVal = v.FreeTypeVar().Union(retVal)
	}
	return retVal
}

func (cs Constraints) Format(state fmt.State, c rune) {
	state.Write([]byte("Constraints["))
	for i, c := range cs {
		if i < len(cs)-1 {
			fmt.Fprintf(state, "%v, ", c)
		} else {
			fmt.Fprintf(state, "%v", c)
		}
	}
	state.Write([]byte{']'})
}

// Types is a slice of Type. Future additions to the methods of this type may be possible
type Types []Type

func (ts Types) Contains(t Type) bool {
	for _, T := range ts {
		if t.Eq(T) {
			return true
		}
	}
	return false
}

// func (ts Types) Apply(sub Subs) Substitutable {
// 	for i, t := range ts {
// 		ts[i] = t.Apply(sub).(Type)
// 	}
// 	return ts
// }

// func (ts Types) FreeTypeVar() TypeVarSet {
// 	var retVal TypeVarSet
// 	for _, v := range ts {
// 		retVal = v.FreeTypeVar().Union(retVal)
// 	}
// 	return retVal
// }
