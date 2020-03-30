// +build debug

package hm

import (
	"fmt"
	"log"
	"os"
	"strings"
	"sync/atomic"
)

// DEBUG returns true when it's in debug mode
const DEBUG = false

var tabcount uint32

var _logger_ = log.New(os.Stderr, "", 0)
var replacement = "\n"

func tc() int {
	return int(atomic.LoadUint32(&tabcount))
}

func enterLoggingContext() {
	atomic.AddUint32(&tabcount, 1)
	tabs := tc()
	_logger_.SetPrefix(strings.Repeat("\t", tabs))
	replacement = "\n" + strings.Repeat("\t", tabs)
}

func leaveLoggingContext() {
	tabs := tc()
	tabs--

	if tabs < 0 {
		atomic.StoreUint32(&tabcount, 0)
		tabs = 0
	} else {
		atomic.StoreUint32(&tabcount, uint32(tabs))
	}
	_logger_.SetPrefix(strings.Repeat("\t", tabs))
	replacement = "\n" + strings.Repeat("\t", tabs)
}

func logf(format string, others ...interface{}) {
	if DEBUG {
		// format = strings.Replace(format, "\n", replacement, -1)
		s := fmt.Sprintf(format, others...)
		s = strings.Replace(s, "\n", replacement, -1)
		_logger_.Println(s)
		// _logger_.Printf(format, others...)
	}
}
