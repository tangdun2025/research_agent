from enum import Enum
import threading
from typing import Any, Dict, Type

from pydantic_core.core_schema import date_schema, model_field

class LLMTypes(Enum):
    """LLM类型枚举"""
    LANGUAGE = "language"      # 基础语言模型
    REASONING = "reasoning"    # 逻辑推理模型
    MULTIMODAL = "multimodal"  # 多模态模型
    CUSTOM = "custom"          # 自定义模型
    DEEPSEEK = "deepseek"  # 新增DeepSeek类型

class LLMBase:
    """LLM基类（抽象接口）"""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    def call(self, input_data: Any) -> Any:
        """统一调用接口"""
        raise NotImplementedError

class LanguageLLM(LLMBase):
    """语言模型实现"""
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        from openai import OpenAI
        self.client = OpenAI(
                        api_key=config["api_key"]
                        , base_url=config["base_url"]
                        # , model_name=config["model_name"]
        )

    def call(self, input_data: str) -> str:
        # 实际调用语言模型的代码
        # return f"Language response: {input_data}"
        response = self.client.chat.completions.create(
                            model=self.config["model_name"],
                            messages=[
                                {"role": "system", "content": "You are a helpful assistant"},
                                {"role": "user", "content":input_data},
                            ],
                            stream=False
        )
        # 检查response是否有效并返回内容，如果为None则返回空字符串
        return response.choices[0].message.content if response.choices[0].message.content is not None else ""

class ReasoningLLM(LLMBase):
    """推理模型实现"""
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        from openai import OpenAI
        self.client = OpenAI(
                        api_key=config["api_key"]
                        , base_url=config["base_url"]
                        # , model_name=config["model_name"]
    )

    def call(self, input_data: str) -> str:
        # 实际调用推理模型的代码
        # return f"Reasoning result: {input_data}"
        response = self.client.chat.completions.create(
                            model=self.config["model_name"],
                            messages=[
                                {"role": "system", "content": "You are a helpful assistant"},
                                {"role": "user", "content":input_data},
                            ],
                            stream=False
        )
        # 检查response是否有效并返回内容，如果为None则返回空字符串
        return response.choices[0].message.content if response.choices[0].message.content is not None else ""

class MultimodalLLM(LLMBase):
    """多模态模型实现"""
    def call(self, input_data: Dict) -> Dict:
        # 处理多模态输入输出
        return {"image": "processed", "text": input_data.get("text")}

class DeepSeekLLM(LLMBase):
    """DeepSeek官方SDK集成实现"""
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        from openai import OpenAI
        self.client = OpenAI(
                        api_key=config["api_key"]
                        , base_url=config["base_url"]
                        # , model_name=config["model_name"]
            )
    def call(self, input_data: str) -> str:
        # 调用官方SDK接口
        response = self.client.chat.completions.create(
                            model=self.config["model_name"],
                            messages=[
                                {"role": "system", "content": "You are a helpful assistant"},
                                {"role": "user", "content":input_data},
                            ],
                            stream=False
        )
        # 检查response是否有效并返回内容，如果为None则返回空字符串
        return response.choices[0].message.content if response.choices[0].message.content is not None else ""

import os
import yaml
from pathlib import Path

class LLMContainer:
    """LLM容器管理单例类"""
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if not cls._instance:
                cls._instance = super().__new__(cls)
                cls._instance._init_container()
            return cls._instance
            
    def _init_container(self):
        """从配置文件加载配置"""
        config_path = Path(__file__).parent.parent / "config" / "llm_config.yaml"
        
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
                self._load_configurations(config_data)
        except FileNotFoundError:
            raise RuntimeError(f"LLM config file not found at {config_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading LLM config: {str(e)}")
            
        self._registry = {
            LLMTypes.LANGUAGE: LanguageLLM,
            LLMTypes.REASONING: ReasoningLLM,
            LLMTypes.MULTIMODAL: MultimodalLLM,
            LLMTypes.DEEPSEEK: DeepSeekLLM  # 注册DeepSeek实现
        }
        self._instances = {}

    def _load_configurations(self, config_data: dict):
        """解析配置文件"""
        self._configs = {
            LLMTypes.LANGUAGE: self._process_config(config_data['llm_configurations']['language']),
            LLMTypes.REASONING: self._process_config(config_data['llm_configurations']['reasoning']),
            LLMTypes.MULTIMODAL: self._process_config(config_data['llm_configurations']['multimodal']),
            LLMTypes.DEEPSEEK: self._process_config(config_data['llm_configurations']['deepseek'])  # 加载DeepSeek配置
        }

    def _process_config(self, config: dict) -> dict:
        """处理环境变量替换"""
        processed = {}
        for k, v in config.items():
            if isinstance(v, str) and v.startswith("${") and v.endswith("}"):
                env_var = v[2:-1]
                processed[k] = os.getenv(env_var, "")
            else:
                processed[k] = v
        return processed
    
    def register_llm(self, 
                    llm_type: LLMTypes, 
                    llm_class: Type[LLMBase],
                    config: Dict[str, Any]):
        """注册自定义LLM"""
        if not issubclass(llm_class, LLMBase):
            raise ValueError("LLM class must inherit from LLMBase")
        self._registry[llm_type] = llm_class
        self._configs[llm_type] = config
    
    def get_instance(self, llm_type: LLMTypes) -> LLMBase:
        """获取LLM单例"""
        if llm_type not in self._registry:
            raise KeyError(f"LLM type {llm_type} not registered")
            
        if llm_type not in self._instances:
            config = self._configs[llm_type]
            llm_class = self._registry[llm_type]
            self._instances[llm_type] = llm_class(config)
            
        return self._instances[llm_type]

# 示例用法
if __name__ == "__main__":
    # 获取容器实例
    container = LLMContainer()
    
    # 获取语言模型
    lang_llm = container.get_instance(LLMTypes.LANGUAGE)
    print('语言模型')
    print(lang_llm.call("你是谁"))
    
    # 获取推理模型
    reason_llm = container.get_instance(LLMTypes.REASONING)
    print('推理模型')
    print(reason_llm.call("你是谁"))

    # 注册自定义模型
    class CustomLLM(LLMBase):
        def call(self, input_data: str) -> str:
            return f"Custom: {input_data}"
    
    container.register_llm(
        LLMTypes.CUSTOM,
        CustomLLM,
        {"model": "custom-v1"}
    )
    
    # 使用自定义模型
    while True:
        user_input = input("Enter a query (or 'q' to quit): ")
        if user_input.lower() == 'q':
            break
        custom_llm = container.get_instance(LLMTypes.DEEPSEEK)
        print(custom_llm.call(user_input))
    # ds_llm = container.get_instance(LLMTypes.DEEPSEEK)
    # query = input("请输入您的问题：")
    # print(ds_llm.call(query))