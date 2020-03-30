package hm

import "fmt"

// Scheme represents a polytype.
// It basically says this:
//		∀TypeVariables.Type.
// What this means is for all TypeVariables enclosed in Type, those TypeVariables can be of any Type.
type Scheme struct {
	tvs TypeVarSet
	t   Type
}

func NewScheme(tvs TypeVarSet, t Type) *Scheme {
	return &Scheme{
		tvs: tvs,
		t:   t,
	}
}

func (s *Scheme) Apply(sub Subs) Substitutable {
	logf("s: %v, sub: %v", s, sub)
	if sub == nil {
		return s
	}
	sub = sub.Clone()
	defer ReturnSubs(sub)

	for _, tv := range s.tvs {
		sub = sub.Remove(tv)
	}

	s.t = s.t.Apply(sub).(Type)
	return s
}

func (s *Scheme) FreeTypeVar() TypeVarSet {
	ftvs := s.t.FreeTypeVar()
	tvs := s.tvs.Set()
	return ftvs.Difference(tvs)
}

func (s *Scheme) Clone() *Scheme {
	tvs := make(TypeVarSet, len(s.tvs))
	for i, v := range s.tvs {
		tvs[i] = v
	}
	return &Scheme{
		tvs: tvs,
		t:   s.t,
	}
}

func (s *Scheme) Format(state fmt.State, c rune) {
	state.Write([]byte("∀["))
	for i, tv := range s.tvs {
		if i < len(s.tvs)-1 {
			fmt.Fprintf(state, "%v, ", tv)
		} else {
			fmt.Fprintf(state, "%v", tv)
		}
	}
	fmt.Fprintf(state, "]: %v", s.t)
}

// Type returns the type of the scheme, as well as a boolean indicating if *Scheme represents a monotype. If it's a polytype, it'll return false
func (s *Scheme) Type() (t Type, isMonoType bool) {
	if len(s.tvs) == 0 {
		return s.t, true
	}
	return s.t, false
}

// Normalize normalizes the type variables in  a scheme, so all the names will be in alphabetical order
func (s *Scheme) Normalize() (err error) {
	tfv := s.t.FreeTypeVar()

	if len(tfv) == 0 {
		return nil
	}

	defer ReturnTypeVarSet(tfv)
	ord := BorrowTypeVarSet(len(tfv))
	for i := range tfv {
		ord[i] = TypeVariable(letters[i])
	}

	s.t, err = s.t.Normalize(tfv, ord)
	s.tvs = ord.Set()
	return
}
