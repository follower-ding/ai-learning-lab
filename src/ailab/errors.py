"""课程里可预期的失败。调用方要能区分它们和程序写错。"""


class LabError(Exception):
    """学习实验中的预期失败。"""


class JsonExtractError(LabError):
    """模型输出里没有可用的 JSON 对象，或字段不在约定内。"""


class VectorDimensionError(LabError):
    """查询向量和库里的向量长度不一致。"""


class ToolArgumentError(LabError):
    """工具参数缺失、类型不对，或超出允许范围。"""


class ConfigError(LabError):
    """真实接口的配置或返回结构不能用。"""
