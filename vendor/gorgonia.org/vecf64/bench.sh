set -ex

benchtime=${1:-1s}

go test -bench . -benchtime $benchtime
go test -tags='sse' -bench . -benchtime $benchtime
go test -tags='avx' -bench . -benchtime $benchtime

# travis compiles commands in script and then executes in bash.  By adding
# set -e we are changing the travis build script's behavior, and the set
# -e lives on past the commands we are providing it.  Some of the travis
# commands are supposed to exit with non zero status, but then continue
# executing.  set -x makes the travis log files extremely verbose and
# difficult to understand.
#
# see travis-ci/travis-ci#5120
set +ex
