"""pytest 全局设置。

``agent_study.config`` 在 import 时就会实例化 ``Settings``，而 pydantic-settings
的优先级是「环境变量 > .env 文件」。这里写死一组假值，挡住本机真实配置，
让测试既能离线跑，读到的配置也是确定的。
"""
import os

os.environ["API_KEY"] = "test-api-key"
os.environ["MODEL_NAME"] = "MiniMax-M3.1-Flash"
os.environ["API_BASE_URL"] = "https://api.minimax.io/v1"
os.environ["ANTHROPIC_BASE_URL"] = "https://api.minimax.io/anthropic"
