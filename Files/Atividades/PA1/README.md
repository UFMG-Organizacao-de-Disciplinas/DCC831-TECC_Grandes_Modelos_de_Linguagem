# Warnings

## uv error

Failed to hardlink files; falling back to full copy. This may lead to degraded performance. If the cache and target directories are on different filesystems, hardlinking may not be supported. If this is intentional, set `export UV_LINK_MODE=copy` or use `--link-mode=copy` to suppress this warning.

## `python_environment_check`

(2025-09-15 17:15:48.756642: I tensorflow/core/util/port.cc:153) oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
