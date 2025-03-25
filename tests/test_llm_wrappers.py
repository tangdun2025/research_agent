
import sys
import os
# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import patch
import unittest

from llm import *

class TestLLMWrappers(unittest.TestCase):
    def setUp(self):
        self.language_wrapper = LanguageLLMWrapper()
        self.reasoning_wrapper = ReasoningLLMWrapper()
        self.deepseek_wrapper = DeepSeekLLMWrapper()

    def test_base_wrapper_properties(self):
        """测试基类属性"""
        self.assertIsInstance(self.language_wrapper, BaseLLMWrapper)
        self.assertEqual(self.reasoning_wrapper._llm_type, 'languagellm')

    def test_language_llm_response(self):   
        """测试语言模型正常响应"""
        response = self.language_wrapper._call("Hello")
        self.assertIn("Language response", response)

    def test_reasoning_llm_response(self):
        """测试推理模型正常响应"""
        response = self.reasoning_wrapper._call("1+1=?")
        self.assertIn("Reasoning result", response)

    @patch('llm.langchain_adapter.LLMContainer')
    def test_deepseek_api_call(self, mock_container):
        """测试DeepSeek API调用流程"""
        # 配置mock
        mock_instance = mock_container.return_value.get_instance.return_value
        mock_instance.call.return_value = "Mocked response"
        
        # 执行调用
        result = self.deepseek_wrapper._call("Test")
        
        # 验证调用链
        mock_container.return_value.get_instance.assert_called_once()
        mock_instance.call.assert_called_once_with("Test")
        self.assertEqual(result, "Mocked response")

    # def test_input_validation(self):
    #     """测试多模态输入类型验证"""
    #     with self.assertRaises(TypeError):
    #         # 测试错误类型输入
    #         wrapper = MultimodalLLMWrapper()
    #         wrapper._call("invalid input type")

if __name__ == '__main__':
    unittest.main()