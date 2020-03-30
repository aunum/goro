package hm

// A Namer is anything that knows its own name
type Namer interface {
	Name() string
}

// A Typer is an Expression node that knows its own Type
type Typer interface {
	Type() Type
}

// An Inferer is an Expression that can infer its own Type given an Env
type Inferer interface {
	Infer(Env, Fresher) (Type, error)
}

// An Expression is basically an AST node. In its simplest form, it's lambda calculus
type Expression interface {
	Body() Expression
}

// Var is an expression representing a variable
type Var interface {
	Expression
	Namer
	Typer
}

// Literal is an Expression/AST Node representing a literal
type Literal interface {
	Var
	IsLit() bool
}

// Apply is an Expression/AST node that represents a function application
type Apply interface {
	Expression
	Fn() Expression
}

// LetRec is an Expression/AST node that represents a recursive let
type LetRec interface {
	Let
	IsRecursive() bool
}

// Let is an Expression/AST node that represents the standard let polymorphism found in functional languages
type Let interface {
	// let name = def in body
	Expression
	Namer
	Def() Expression
}

// Lambda is an Expression/AST node that represents a function definiton
type Lambda interface {
	Expression
	Namer
	IsLambda() bool
}
