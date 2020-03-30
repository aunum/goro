package hm

// An Env is essentially a map of names to schemes
type Env interface {
	Substitutable
	SchemeOf(string) (*Scheme, bool)
	Clone() Env

	Add(string, *Scheme) Env
	Remove(string) Env
}

type SimpleEnv map[string]*Scheme

func (e SimpleEnv) Apply(sub Subs) Substitutable {
	logf("Applying %v to env", sub)
	if sub == nil {
		return e
	}

	for _, v := range e {
		v.Apply(sub) // apply mutates Scheme, so no need to set
	}
	return e
}

func (e SimpleEnv) FreeTypeVar() TypeVarSet {
	var retVal TypeVarSet
	for _, v := range e {
		retVal = v.FreeTypeVar().Union(retVal)
	}
	return retVal
}

func (e SimpleEnv) SchemeOf(name string) (retVal *Scheme, ok bool) { retVal, ok = e[name]; return }
func (e SimpleEnv) Clone() Env {
	retVal := make(SimpleEnv)
	for k, v := range e {
		retVal[k] = v.Clone()
	}
	return retVal
}

func (e SimpleEnv) Add(name string, s *Scheme) Env {
	e[name] = s
	return e
}

func (e SimpleEnv) Remove(name string) Env {
	delete(e, name)
	return e
}
