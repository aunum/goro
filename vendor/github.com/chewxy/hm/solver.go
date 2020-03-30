package hm

type solver struct {
	sub Subs
	err error
}

func newSolver() *solver {
	return new(solver)
}

func (s *solver) solve(cs Constraints) {
	logf("solving constraints: %d", len(cs))
	enterLoggingContext()
	defer leaveLoggingContext()
	logf("starting sub %v", s.sub)
	if s.err != nil {
		return
	}

	switch len(cs) {
	case 0:
		return
	default:
		var sub Subs
		c := cs[0]
		sub, s.err = Unify(c.a, c.b)
		defer ReturnSubs(s.sub)

		s.sub = compose(sub, s.sub)
		cs = cs[1:].Apply(s.sub).(Constraints)
		s.solve(cs)

	}
	logf("Ending Sub %v", s.sub)
	return
}
