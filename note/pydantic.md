# Pydantic 学习笔记

## 模型实例化

- model 生成 instance 只接受关键字参数，不能用位置参数。
- model 可以直接接受 dict 类型自动解析：`user_1 = User(**data)`。
- `@classmethod` 相当于类的构造器。

## 可变默认值

pydantic 对 class 中可变默认值是安全的——每个实例会拿到独立的副本，不会像 dataclass 那样共享同一个列表（这正是 dataclass 需要 `field(default_factory=list)` 的原因，pydantic 帮你处理了）。

## 序列化方法

- `model_dump` 方法：class -> dict
- `model_dump_json` 方法：class -> json

## 加载不同环境变量

settings 加载不同 env 方式：

```bash
APP_ENV=prod uv run python -m agent_study   # Linux/Git Bash
set APP_ENV=prod && uv run python -m agent_study  # Windows cmd
$env:APP_ENV="prod"; uv run python -m agent_study # PowerShell
```
