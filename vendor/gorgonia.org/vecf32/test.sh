set -ex

go env

go test -v -a -coverprofile=test.cover
go test -tags='sse' -v -a -coverprofile=test.cover.sse
go test -tags='avx' -v -a -coverprofile=test.cover.avx

echo "mode: set" > final.cover

tail -q -n +2 test.cover test.cover.sse test.cover.avx >> ./final.cover
goveralls -coverprofile=./final.cover -service=travis-ci

# travis compiles commands in script and then executes in bash.  By adding
# set -e we are changing the travis build script's behavior, and the set
# -e lives on past the commands we are providing it.  Some of the travis
# commands are supposed to exit with non zero status, but then continue
# executing.  set -x makes the travis log files extremely verbose and
# difficult to understand.
# 
# see travis-ci/travis-ci#5120
set +ex
