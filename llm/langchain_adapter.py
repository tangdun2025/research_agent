from langchain_core.language_models import BaseLLM
from typing import Any, Dict, Optional
from .llm_manager import LLMContainer, LLMTypes  # 修正为正确的相对导入
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.outputs import LLMResult, Generation

class BaseLLMWrapper(BaseLLM):
    """通用LLM包装器基类"""
    llm: Optional[Any] = None  # 显式声明字段
    
    def __init__(self, llm_type: LLMTypes):
        super().__init__()  # 必须首先调用父类初始化
        self.llm = LLMContainer().get_instance(llm_type)
        
    @property
    def _llm_type(self) -> str:
        return self.llm.__class__.__name__.lower()

    def _call(self, prompt: str, **kwargs: Any) -> str:
        # 检查llm实例是否存在并调用其方法
        if not self.llm:
            raise ValueError("LLM instance not initialized")
        if hasattr(self.llm, 'call'):
            return self.llm.call(prompt)
        elif hasattr(self.llm, 'complete'):
            return self.llm.complete(prompt)
        else:
            raise AttributeError("LLM instance does not have 'call' or 'complete' method")


    def _generate(
        self,
        prompts: list[str],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """批量生成实现"""
        # generations = []
        # for prompt in prompts:
        #     try:
        #         # 调用底层LLM并处理停止词
        #         response = self.llm.call(prompt)
        #         if stop:
        #             for stop_word in stop:
        #                 response = response.split(stop_word)[0]
        #         generations.append([Generation(text=response)])
        #     except Exception as e:
        #         if run_manager:
        #             run_manager.on_llm_error(e)
        #         raise

        return LLMResult(generations=[[Generation(text="")]], llm_output={})

class LanguageLLMWrapper(BaseLLMWrapper):
    """语言模型包装器"""
    def __init__(self):
        super().__init__(LLMTypes.LANGUAGE)

class ReasoningLLMWrapper(BaseLLMWrapper):
    """推理模型包装器"""
    def __init__(self):
        super().__init__(LLMTypes.REASONING)


class DeepSeekLLMWrapper(BaseLLMWrapper):
    """DeepSeek模型包装器"""
    def __init__(self):
        super().__init__(LLMTypes.DEEPSEEK)


if __name__ == "__main__":
    wrapper = DeepSeekLLMWrapper()
    print(wrapper._llm_type)
    print(wrapper._call("你好"))