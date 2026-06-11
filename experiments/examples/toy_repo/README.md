# Toy Repo 规则

该目录是 toy patch pilot 使用的极小示例仓库。

规则：

- ingestion timestamps 可以使用末尾 `Z`，按 UTC 处理。
- API paths 必须标准化为一个前导斜杠。
- 只对 transient HTTP status codes 做 retry。
