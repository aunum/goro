package log

import (
	"fmt"
	"strings"

	"github.com/fatih/color"
)

// Logger is a logger.
type Logger struct {
	// Level to log at. Defaults to info level.
	Level Level

	// Color should be enabled for logs.
	Color bool

	// TestMode is enabled.
	TestMode bool

	// Timestamps should be printed.
	Timestamps bool
}

// DefaultLogger is the default logger.
var DefaultLogger = &Logger{
	Level: InfoLevel,
	Color: true,
}

// NewLogger is a new logger.
func NewLogger(level Level, color bool) *Logger {
	return &Logger{
		Level: level,
		Color: color,
	}
}

// Fatalf message logs formatted Error then exits with code 1.
func (l *Logger) Fatalf(format string, a ...interface{}) {
	if l.Level >= ErrorLevel {
		fatalf(l.Timestamps, l.Color, format, a...)
	}
}

// Fataly prints the YAML represtation of an object at Error level then exits with code 1.
func (l *Logger) Fataly(name string, obj interface{}) {
	yam, err := SPrintYAML(obj)
	if err != nil {
		Error(err)
		Fatal(obj)
		return
	}
	l.Fatalf("%s >> \n\n%s\n", name, yam)
}

// Fatalv prints values in a k:v fromat and then exists with code 1.
func (l *Logger) Fatalv(v ...interface{}) {
	out := []string{}
	for _, value := range v {
		yam, err := SPrintYAML(value)
		if err != nil {
			Error(err)
			Fatal(value)
			return
		}
		out = append(out, yam)
	}
	l.Fatal(strings.Join(out, ","))
}

// Fatal logs Error message then exits with code 1.
func (l *Logger) Fatal(a ...interface{}) {
	l.Fatalf(buildFormat(a...), a...)
}

// Errorf is a formatted Error message.
func (l *Logger) Errorf(format string, a ...interface{}) {
	if l.Level >= ErrorLevel {
		errorf(l.Timestamps, l.Color, format, a...)
	}
}

// Errorv prints value in a k:v fromat.
func (l *Logger) Errorv(name string, value interface{}) {
	l.Errorf("%s: %v", keyf(name), value)
}

// Errorvb prints value in a k:v fromat with the value on a new line.
func (l *Logger) Errorvb(name string, value interface{}) {
	l.Errorf("%s: \n%v\n", keyf(name), value)
}

// Errory prints the YAML represtation of an object at Error level.
func (l *Logger) Errory(name string, obj interface{}) {
	yam, err := SPrintYAML(obj)
	if err != nil {
		Error(err)
		Error(obj)
		return
	}
	l.Errorf("%s >> \n\n%s\n", name, yam)
}

// Error message.
func (l *Logger) Error(a ...interface{}) {
	l.Errorf(buildFormat(a...), a...)
}

// Infof is a formatted Info message.
func (l *Logger) Infof(format string, a ...interface{}) {
	if l.Level >= InfoLevel {
		infof(l.Timestamps, l.Color, format, a...)
	}
}

// Infov prints value in a k:v fromat.
func (l *Logger) Infov(name string, value interface{}) {
	l.Infof("%s: %v", keyf(name), value)
}

// Infovb prints value in a k:v fromat with the value on a new line.
func (l *Logger) Infovb(name string, value interface{}) {
	l.Infof("%s: \n%v\n", keyf(name), value)
}

// Infoy prints the YAML represtation of an object at Info level.
func (l *Logger) Infoy(name string, obj interface{}) {
	yam, err := SPrintYAML(obj)
	if err != nil {
		Error(err)
		Info(obj)
		return
	}
	l.Infof("%s >> \n\n%s\n", name, yam)
}

// Info message.
func (l *Logger) Info(a ...interface{}) {
	l.Infof(buildFormat(a...), a...)
}

// Successf is a formatted Success message.
func (l *Logger) Successf(format string, a ...interface{}) {
	if l.Level >= InfoLevel {
		successf(l.Timestamps, l.Color, format, a...)
	}
}

// Successv prints value in a k:v fromat.
func (l *Logger) Successv(name string, value interface{}) {
	l.Successf("%s: %v", keyf(name), value)
}

// Successvb prints value in a k:v fromat with the value on a new line.
func (l *Logger) Successvb(name string, value interface{}) {
	l.Successf("%s: \n%v\n", keyf(name), value)
}

// Successy prints the YAML represtation of an object at Success level.
func (l *Logger) Successy(name string, obj interface{}) {
	yam, err := SPrintYAML(obj)
	if err != nil {
		Error(err)
		Success(obj)
		return
	}
	l.Successf("%s >> \n\n%s\n", name, yam)
}

// Success message.
func (l *Logger) Success(a ...interface{}) {
	l.Successf(buildFormat(a...), a...)
}

// Debugf is a formatted Debug message.
func (l *Logger) Debugf(format string, a ...interface{}) {
	if l.Level >= DebugLevel {
		debugf(l.Timestamps, l.Color, format, a...)
	}
}

// Debugv prints value in a k:v fromat.
func (l *Logger) Debugv(name string, value interface{}) {
	l.Debugf("%s: %v", keyf(name), value)
}

// Debugvb prints value in a k:v fromat with the value on a new line.
func (l *Logger) Debugvb(name string, value interface{}) {
	l.Debugf("%s: \n%v\n", keyf(name), value)
}

// Debugy prints the YAML represtation of an object at Debug level.
func (l *Logger) Debugy(name string, obj interface{}) {
	yam, err := SPrintYAML(obj)
	if err != nil {
		Error(err)
		Debug(obj)
		return
	}
	l.Debugf("%s >> \n\n%s\n", name, yam)
}

// Debug message.
func (l *Logger) Debug(a ...interface{}) {
	l.Debugf(buildFormat(a...), a...)
}

// Dumpf is a formatted Dump message.
func (l *Logger) Dumpf(format string, a ...interface{}) {
	if l.Level >= DumpLevel {
		dumpf(l.Timestamps, l.Color, format, a...)
	}
}

// Dumpv prints value in a k:v fromat.
func (l *Logger) Dumpv(name string, value interface{}) {
	l.Debugf("%s: %v", keyf(name), value)
}

// Dumpvb prints value in a k:v fromat with the value on a new line.
func (l *Logger) Dumpvb(name string, value interface{}) {
	l.Dumpf("%s: \n%v\n", keyf(name), value)
}

// Dumpy prints the YAML represtation of an object at Dump level.
func (l *Logger) Dumpy(name string, obj interface{}) {
	yam, err := SPrintYAML(obj)
	if err != nil {
		Error(err)
		Dump(obj)
	}
	l.Dumpf("%s >> \n\n%s\n", name, yam)
}

// Dump message.
func (l *Logger) Dump(a ...interface{}) {
	l.Dumpf(buildFormat(a...), a...)
}

// Warningf is a formatted Warning message.
func (l *Logger) Warningf(format string, a ...interface{}) {
	if l.Level >= WarningLevel {
		warningf(l.Timestamps, l.Color, format, a...)
	}
}

// Warningv prints value in a k:v fromat.
func (l *Logger) Warningv(name string, value interface{}) {
	l.Warningf("%s: %v", keyf(name), value)
}

// Warningvb prints value in a k:v fromat with the value on a new line.
func (l *Logger) Warningvb(name string, value interface{}) {
	l.Warningf("%s: \n%v\n", keyf(name), value)
}

// Warningy prints the YAML represtation of an object at Warning level.
func (l *Logger) Warningy(name string, obj interface{}) {
	yam, err := SPrintYAML(obj)
	if err != nil {
		Error(err)
		Warning(obj)
		return
	}
	l.Warningf("%s >> \n\n%s\n", name, yam)
}

// Warning message.
func (l *Logger) Warning(a ...interface{}) {
	l.Warningf(buildFormat(a...), a...)
}

// Break prints a break in the logs.
func (l *Logger) Break() {
	fmt.Println(color.HiMagentaString("-------"))
}

// BreakHard prints a hard break in the logs.
func (l *Logger) BreakHard() {
	fmt.Println(color.CyanString("========"))
}

// BreakStar prints a star break in the logs.
func (l *Logger) BreakStar() {
	fmt.Println(color.RedString("*********"))
}

// BreakPound prints a pound break in the logs.
func (l *Logger) BreakPound() {
	fmt.Println(color.HiBlueString("#########"))
}
