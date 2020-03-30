// Package graph provides helper methods for a Gorgonia graph.
package graph

import (
	"fmt"
	"io/ioutil"
	"os/exec"

	"github.com/aunum/gold/pkg/v1/common/require"
	"github.com/aunum/log"
	"github.com/skratchdot/open-golang/open"

	g "gorgonia.org/gorgonia"
)

// Visualize the graph using graphviz.
//
// Note: this requires graphviz `dot` to be installed on the host os.
func Visualize(graph *g.ExprGraph) {
	f, err := ioutil.TempFile("", "graph.*.dot")
	require.NoError(err)
	_, err = f.Write([]byte(graph.ToDot()))
	require.NoError(err)
	tempPath := f.Name()
	svgPath := fmt.Sprintf("%s.svg", f.Name())
	log.Debug("saved file: ", tempPath)
	cmd := exec.Command("dot", "-Tsvg", tempPath, "-O")
	err = cmd.Run()
	require.NoError(err)
	err = open.Run(svgPath)
	require.NoError(err)
}
