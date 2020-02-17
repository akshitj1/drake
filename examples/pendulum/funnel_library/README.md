```
TARGET=//examples/pendulum/funnel_library:simulate
bazel aquery "deps($TARGET)" -c dbg --spawn_strategy=standalone > /dev/null
```


*debug*
```
bazel build $TARGET -c dbg --spawn_strategy=standalone
lldb bazel-bin/examples/pendulum/funnel_library/simulate
```

*lldb*
```
thread backtrace
setting set target.max-string-summary-length 10000
```