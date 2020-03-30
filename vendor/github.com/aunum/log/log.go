package log

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"runtime/debug"
	"strings"
	"time"

	"github.com/fatih/color"
	yamlconv "github.com/ghodss/yaml"
	"github.com/golang/protobuf/jsonpb"
	"github.com/golang/protobuf/proto"
)

// Level is a log level.
type Level int

const (
	// ErrorLevel logging.
	ErrorLevel Level = 1
	// WarningLevel logging.
	WarningLevel Level = 2
	// InfoLevel logging.
	InfoLevel Level = 3
	// DebugLevel logging.
	DebugLevel Level = 4
	// DumpLevel logging.
	DumpLevel Level = 5
)

const (
	// ErrorLabel is a label for a Error message.
	ErrorLabel = "✖"
	// DebugLabel is a label for a debug message.
	DebugLabel = "▶"
	// DumpLabel is a label for a dump message.
	DumpLabel = "▼"
	// InfoLabel is a label for an informative message.
	InfoLabel = "ℹ"
	// SuccessLabel is a label for a success message.
	SuccessLabel = "✔"
	// WarningLabel is a label for a warning message.
	WarningLabel = "!"
)

var (
	// GlobalLevel to log at. Defaults to info level.
	GlobalLevel = InfoLevel
	// Color should be enabled for logs.
	Color = true
	// TestMode is enabled.
	TestMode = false
	// Timestamps should be printed.
	Timestamps = false
)

// Fatalf message logs formatted Error then exits with code 1.
func Fatalf(format string, a ...interface{}) {
	if GlobalLevel >= ErrorLevel {
		fatalf(Timestamps, Color, format, a...)
	}
}

func fatalf(timestamps, colored bool, format string, a ...interface{}) {
	a, w := extractLoggerArgs(format, a...)
	l := ErrorLabel
	if colored {
		w = color.Output
		l = color.RedString(l)
	}
	s := fmt.Sprintf(label(format, l, timestamps), a...)
	fmt.Fprintf(w, s)
	debug.PrintStack()
	os.Exit(1)
}

// Fataly prints the YAML represtation of an object at Error level then exits with code 1.
func Fataly(name string, obj interface{}) {
	yam, err := SPrintYAML(obj)
	if err != nil {
		Error(err)
		Fatal(obj)
		return
	}
	Fatalf("%s >> \n\n%s\n", name, yam)
}

// Fatalv prints values in a k:v fromat and then exists with code 1.
func Fatalv(v ...interface{}) {
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
	Fatal(strings.Join(out, ","))
}

// Fatal logs Error message then exits with code 1.
func Fatal(a ...interface{}) {
	Fatalf(buildFormat(a...), a...)
}

// Errorf is a formatted Error message.
func Errorf(format string, a ...interface{}) {
	if GlobalLevel >= ErrorLevel {
		errorf(Timestamps, Color, format, a...)
	}
}

func errorf(timestamps, colored bool, format string, a ...interface{}) {
	a, w := extractLoggerArgs(format, a...)
	l := ErrorLabel
	if colored {
		w = color.Output
		l = color.RedString(l)
	}
	s := fmt.Sprintf(label(format, l, timestamps), a...)
	fmt.Fprintf(w, s)
}

// Errorv prints value in a k:v fromat.
func Errorv(name string, value interface{}) {
	Errorf("%s: %v", keyf(name), value)
}

// Errorvb prints value in a k:v fromat with the value on a new line.
func Errorvb(name string, value interface{}) {
	Errorf("%s: \n%v\n", keyf(name), value)
}

// Errory prints the YAML represtation of an object at Error level.
func Errory(name string, obj interface{}) {
	yam, err := SPrintYAML(obj)
	if err != nil {
		Error(err)
		Error(obj)
		return
	}
	Errorf("%s >> \n\n%s\n", name, yam)
}

// Error message.
func Error(a ...interface{}) {
	Errorf(buildFormat(a...), a...)
}

// Infof is a formatted Info message.
func Infof(format string, a ...interface{}) {
	if GlobalLevel >= InfoLevel {
		infof(Timestamps, Color, format, a...)
	}
}

func infof(timestamps, colored bool, format string, a ...interface{}) {
	a, w := extractLoggerArgs(format, a...)
	l := InfoLabel
	if colored {
		w = color.Output
		l = color.CyanString(l)
	}
	s := fmt.Sprintf(label(format, l, timestamps), a...)
	fmt.Fprintf(w, s)
}

// Infov prints value in a k:v fromat.
func Infov(name string, value interface{}) {
	Infof("%s: %v", keyf(name), value)
}

// Infovb prints value in a k:v fromat with the value on a new line.
func Infovb(name string, value interface{}) {
	Infof("%s: \n%v\n", keyf(name), value)
}

// Infoy prints the YAML represtation of an object at Info level.
func Infoy(name string, obj interface{}) {
	yam, err := SPrintYAML(obj)
	if err != nil {
		Error(err)
		Info(obj)
		return
	}
	Infof("%s >> \n\n%s\n", name, yam)
}

// Info message.
func Info(a ...interface{}) {
	Infof(buildFormat(a...), a...)
}

// Successf is a formatted Success message.
func Successf(format string, a ...interface{}) {
	if GlobalLevel >= InfoLevel {
		successf(Timestamps, Color, format, a...)
	}
}

func successf(timestamps, colored bool, format string, a ...interface{}) {
	a, w := extractLoggerArgs(format, a...)
	l := SuccessLabel
	if colored {
		w = color.Output
		l = color.HiGreenString(l)
	}
	s := fmt.Sprintf(label(format, l, timestamps), a...)
	fmt.Fprintf(w, s)
}

// Successv prints value in a k:v fromat.
func Successv(name string, value interface{}) {
	Successf("%s: %v", keyf(name), value)
}

// Successvb prints value in a k:v fromat with the value on a new line.
func Successvb(name string, value interface{}) {
	Successf("%s: \n%v\n", keyf(name), value)
}

// Successy prints the YAML represtation of an object at Success level.
func Successy(name string, obj interface{}) {
	yam, err := SPrintYAML(obj)
	if err != nil {
		Error(err)
		Success(obj)
		return
	}
	Successf("%s >> \n\n%s\n", name, yam)
}

// Success message.
func Success(a ...interface{}) {
	Successf(buildFormat(a...), a...)
}

// Debugf is a formatted Debug message.
func Debugf(format string, a ...interface{}) {
	if GlobalLevel >= DebugLevel {
		debugf(Timestamps, Color, format, a...)
	}
}

func debugf(timestamps, colored bool, format string, a ...interface{}) {
	a, w := extractLoggerArgs(format, a...)
	l := DebugLabel
	if colored {
		w = color.Output
		l = color.MagentaString(l)
	}
	s := fmt.Sprintf(label(format, l, timestamps), a...)
	fmt.Fprintf(w, s)
}

// Debugv prints value in a k:v fromat.
func Debugv(name string, value interface{}) {
	Debugf("%s: %v", keyf(name), value)
}

// Debugvb prints value in a k:v fromat with the value on a new line.
func Debugvb(name string, value interface{}) {
	Debugf("%s: \n%v\n", keyf(name), value)
}

// Debugy prints the YAML represtation of an object at Debug level.
func Debugy(name string, obj interface{}) {
	yam, err := SPrintYAML(obj)
	if err != nil {
		Error(err)
		Debug(obj)
		return
	}
	Debugf("%s >> \n\n%s\n", name, yam)
}

// Debug message.
func Debug(a ...interface{}) {
	Debugf(buildFormat(a...), a...)
}

// Dumpf is a formatted Dump message.
func Dumpf(format string, a ...interface{}) {
	if GlobalLevel >= DumpLevel {
		dumpf(Timestamps, Color, format, a...)
	}
}

func dumpf(timestamps, colored bool, format string, a ...interface{}) {
	a, w := extractLoggerArgs(format, a...)
	l := DumpLabel
	if colored {
		w = color.Output
		l = color.BlueString(l)
	}
	s := fmt.Sprintf(label(format, l, timestamps), a...)
	fmt.Fprintf(w, s)
}

// Dumpv prints value in a k:v fromat.
func Dumpv(name string, value interface{}) {
	Debugf("%s: %v", keyf(name), value)
}

// Dumpvb prints value in a k:v fromat with the value on a new line.
func Dumpvb(name string, value interface{}) {
	Dumpf("%s: \n%v\n", keyf(name), value)
}

// Dumpy prints the YAML represtation of an object at Dump level.
func Dumpy(name string, obj interface{}) {
	yam, err := SPrintYAML(obj)
	if err != nil {
		Error(err)
		Dump(obj)
	}
	Dumpf("%s >> \n\n%s\n", name, yam)
}

// Dump message.
func Dump(a ...interface{}) {
	Dumpf(buildFormat(a...), a...)
}

// Warningf is a formatted Warning message.
func Warningf(format string, a ...interface{}) {
	if GlobalLevel >= WarningLevel {
		warningf(Timestamps, Color, format, a...)
	}
}

func warningf(timestamps, colored bool, format string, a ...interface{}) {
	a, w := extractLoggerArgs(format, a...)
	l := WarningLabel
	if colored {
		w = color.Output
		l = color.HiYellowString(l)
	}
	s := fmt.Sprintf(label(format, l, timestamps), a...)
	fmt.Fprintf(w, s)
}

// Warningv prints value in a k:v fromat.
func Warningv(name string, value interface{}) {
	Warningf("%s: %v", keyf(name), value)
}

// Warningvb prints value in a k:v fromat with the value on a new line.
func Warningvb(name string, value interface{}) {
	Warningf("%s: \n%v\n", keyf(name), value)
}

// Warningy prints the YAML represtation of an object at Warning level.
func Warningy(name string, obj interface{}) {
	yam, err := SPrintYAML(obj)
	if err != nil {
		Error(err)
		Warning(obj)
		return
	}
	Warningf("%s >> \n\n%s\n", name, yam)
}

// Warning message.
func Warning(a ...interface{}) {
	Warningf(buildFormat(a...), a...)
}

// Break prints a break in the logs.
func Break() {
	fmt.Println(color.HiMagentaString("-------"))
}

// BreakHard prints a hard break in the logs.
func BreakHard() {
	fmt.Println(color.CyanString("========"))
}

// BreakStar prints a star break in the logs.
func BreakStar() {
	fmt.Println(color.RedString("*********"))
}

// BreakPound prints a pound break in the logs.
func BreakPound() {
	fmt.Println(color.HiBlueString("#########"))
}

func keyf(key string) string {
	return color.HiBlueString(key)
}

// SPrintYAML returns a YAML string for an object and has support for proto messages.
func SPrintYAML(a interface{}) (string, error) {
	b, err := MarshalJSON(a)
	// doing yaml this way because at times you have nested proto structs
	// that need to be cleaned.
	yam, err := yamlconv.JSONToYAML(b)
	if err != nil {
		return "", err
	}
	return string(yam), nil
}

// MarshalJSON marshals either a proto message or any other interface.
func MarshalJSON(a interface{}) (b []byte, err error) {
	if m, ok := a.(proto.Message); ok {
		marshaller := &jsonpb.Marshaler{}
		var buf bytes.Buffer
		err = marshaller.Marshal(&buf, m)
		if err != nil {
			return
		}
		b = buf.Bytes()
	} else {
		b, err = json.Marshal(a)
		if err != nil {
			return
		}
	}
	return
}

// PrintYAML prints the YAML string of an object and has support for proto messages.
func PrintYAML(a interface{}) error {
	s, err := SPrintYAML(a)
	if err != nil {
		return err
	}
	fmt.Println(s)
	return nil
}

// Check if an error is nil otherwise log fatal.
func Check(err error) {
	if err != nil {
		Fatal(err)
	}
}

func extractLoggerArgs(format string, a ...interface{}) ([]interface{}, io.Writer) {
	var w io.Writer = os.Stdout

	if n := len(a); n > 0 {
		// extract an io.Writer at the end of a
		if value, ok := a[n-1].(io.Writer); ok {
			w = value
			a = a[0 : n-1]
		}
	}

	return a, w
}

func label(format, label string, timestamps bool) string {
	if timestamps {
		return labelWithTime(format, label)
	}
	return labelWithoutTime(format, label)
}

func labelWithTime(format, label string) string {
	t := time.Now()
	rfct := t.Format(time.RFC3339)
	if !strings.Contains(format, "\n") {
		format = fmt.Sprintf("%s%s", format, "\n")
	}
	return fmt.Sprintf("%s [%s]  %s", rfct, label, format)
}

func labelWithoutTime(format, label string) string {
	if !strings.Contains(format, "\n") {
		format = fmt.Sprintf("%s%s", format, "\n")
	}
	return fmt.Sprintf("%s  %s", label, format)
}

func buildFormat(f ...interface{}) string {
	var fin string
	for _, i := range f {
		if _, ok := i.(error); ok {
			fin += "%s "
		} else if _, ok := i.(string); ok {
			fin += "%s "
		} else {
			fin += "%#v "
		}
	}
	return fin
}
