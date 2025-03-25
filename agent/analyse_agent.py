import os
import sys
import json
import logging
import datetime
import re
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain.prompts import PromptTemplate
from langchain_community.llms import Tongyi
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AnalyseAgent")

@dataclass
class ModuleContent:
    """模块内容"""
    title: str
    content: str
    references: List[str]

@dataclass
class AnalysisResult:
    """分析结果"""
    topic: str
    summary: str
    modules: List[ModuleContent]

class AnalyseAgent:
    """内容分析Agent"""
    
    def __init__(self, tongyi_api_key: Optional[str] = None):
        """初始化Agent"""
        # 初始化通义千问LLM
        self.llm = Tongyi(
            model_name="qwen-max",
            api_key=tongyi_api_key or os.environ.get("DASHSCOPE_API_KEY", "sk-xxxxx"),
            temperature=0.7
        )
        
        # 设置文本分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            chunk_overlap=200,
            length_function=len,
        )
        
        # 设置上下文窗口大小限制
        self.max_context_size = 8000
        self.max_module_content_length = 2000
        self.min_module_content_length = 1000
    
    def _read_document(self, document_path: str) -> List[Dict[str, Any]]:
        """读取文档内容并返回结构化数据"""
        logger.info(f"读取文档: {document_path}")
        
        # 清理文件路径，移除可能的额外内容
        document_path = document_path.split('\n')[0].strip()
        
        try:
            with open(document_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 提取文章数据
            articles = []
            for item in data.get('articles', []):
                article = {
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "chunks": item.get("chunks", [])
                }
                articles.append(article)
            
            logger.info(f"文档读取完成，共{len(articles)}篇文章")
            return articles
        except Exception as e:
            logger.error(f"读取文档失败: {e}")
            return []
    
    def _clean_content(self, content: str) -> str:
        """清理文档内容，去除无效内容"""
        logger.info("清理文档内容")
        
        # 移除PDF格式的无效内容
        cleaned_content = re.sub(r'%PDF-\d+\.\d+.*?endobj', '', content, flags=re.DOTALL)
        cleaned_content = re.sub(r'<>stream.*?endstream', '', cleaned_content, flags=re.DOTALL)
        cleaned_content = re.sub(r'<>/ExtGState.*?endobj', '', cleaned_content, flags=re.DOTALL)
        
        # 移除其他无效内容
        cleaned_content = re.sub(r'\r\n\d+ \d+ obj.*?endobj', '', cleaned_content, flags=re.DOTALL)
        
        # 如果清理后内容太少，可能是过度清理，返回原始内容
        if len(cleaned_content) < len(content) * 0.1:
            logger.warning("清理可能过度，返回原始内容")
            return content
        
        logger.info("内容清理完成")
        return cleaned_content
    
    def _extract_key_information(self, articles: List[Dict[str, Any]], topic: str) -> str:
        """从文章中提取与主题相关的关键信息"""
        logger.info(f"提取关键信息，主题: {topic}")
        
        # 合并所有文章内容
        all_content = []
        for article in articles:
            title = article["title"]
            url = article["url"]
            for chunk in article["chunks"]:
                all_content.append(f"标题: {title}\n来源: {url}\n内容: {chunk}")
        
        # 将内容分成较小的批次
        content_batches = []
        current_batch = []
        current_length = 0
        max_batch_size = 7000  # 每批最大字符数
        
        for item in all_content:
            if current_length + len(item) > max_batch_size:
                if current_batch:  # 确保批次不为空
                    content_batches.append("\n\n---\n\n".join(current_batch))
                current_batch = [item]
                current_length = len(item)
            else:
                current_batch.append(item)
                current_length += len(item)
        
        # 添加最后一批
        if current_batch:
            content_batches.append("\n\n---\n\n".join(current_batch))
        
        logger.info(f"内容已分为{len(content_batches)}个批次进行处理")
        
        # 对每个批次提取关键信息
        key_info_batches = []
        
        for i, batch in enumerate(content_batches):
            logger.info(f"处理批次 {i+1}/{len(content_batches)}")
            
            template = """
            请从以下内容中提取与主题"{topic}"相关的关键信息。
            
            内容:
            {content}
            
            请提取所有相关的事实、数据、观点和见解，确保不遗漏重要信息。
            格式要求:
            1. 提取的信息应该是客观的，基于文本内容
            2. 保留原文中的数据和事实
            3. 按照重要性排序
            4. 使用简洁明了的语言
            
            提取的关键信息:
            """
            
            prompt = PromptTemplate(
                template=template,
                input_variables=["topic", "content"]
            )
            
            try:
                chain = prompt | self.llm
                key_info = chain.invoke({"topic": topic, "content": batch})
                key_info_batches.append(key_info)
                # 避免API限流
                time.sleep(1)
            except Exception as e:
                logger.error(f"提取批次{i+1}关键信息失败: {e}")
                # 如果失败，添加一个空字符串，保持批次数量一致
                key_info_batches.append("")
        
        # 合并所有批次的关键信息
        combined_key_info = "\n\n".join(key_info_batches)
        
        logger.info("关键信息提取完成")
        return combined_key_info
    
    def _identify_modules(self, key_info: str, topic: str) -> List[str]:
        """根据关键信息和主题识别4-8个模块"""
        logger.info("识别内容模块")
        
        template = """
        请根据以下关键信息，为主题"{topic}"划分4-8个内容模块。
        
        关键信息:
        {key_info}
        
        请根据信息的逻辑关系和主题相关性，划分4-8个内容模块。每个模块应该有明确的主题和焦点。
        
        输出格式:
        1. 模块一标题
        2. 模块二标题
        3. 模块三标题
        ...
        
        模块划分:
        """
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["topic", "key_info"]
        )
        
        try:
            # 如果关键信息太长，截取前10000个字符
            # if len(key_info) > 10000:
            #     logger.warning(f"关键信息过长 ({len(key_info)} 字符)，截取前10000字符")
            #     key_info = key_info[:10000] + "...(内容已截断)"
            #             # 如果关键信息太长，进行智能截取
            if len(key_info) > 10000:
                logger.warning(f"关键信息过长 ({len(key_info)} 字符)，进行智能截取")
                
                # 使用文本分割器将内容分成较小的块
                chunks = self.text_splitter.split_text(key_info)
                
                # 选择重要的块：开头、中间和结尾的部分
                selected_chunks = []
                
                # 添加开头部分（前2个块或30%，取较小值）
                head_count = min(2, max(1, int(len(chunks) * 0.3)))
                selected_chunks.extend(chunks[:head_count])
                
                # 添加中间部分（随机选择2个块或20%，取较小值）
                if len(chunks) > head_count + 2:
                    mid_count = min(2, max(1, int(len(chunks) * 0.2)))
                    mid_indices = [int(len(chunks) * 0.5) + i for i in range(mid_count)]
                    for idx in mid_indices:
                        if 0 <= idx < len(chunks) and idx >= head_count:
                            selected_chunks.append(chunks[idx])
                
                # 添加结尾部分（最后2个块或30%，取较小值）
                tail_count = min(2, max(1, int(len(chunks) * 0.3)))
                selected_chunks.extend(chunks[-tail_count:])
                
                # 合并选定的块，并添加说明
                key_info = "\n\n[...内容已智能截取...]\n\n".join(selected_chunks)
                key_info += "\n\n[注：由于内容过长，已进行智能截取，保留了开头、中间和结尾的重要部分]"
                
                logger.info(f"智能截取后的内容长度: {len(key_info)} 字符")
            

            chain = prompt | self.llm
            modules_text = chain.invoke({"topic": topic, "key_info": key_info})
            
            # 解析模块标题
            modules = []
            for line in modules_text.split('\n'):
                # 匹配形如 "1. 标题" 或 "1、标题" 的行
                match = re.match(r'^\s*\d+[\.\、]\s*(.*?)$', line)
                if match:
                    modules.append(match.group(1).strip())
            
            # 确保至少有4个模块，最多8个模块
            if len(modules) < 4:
                logger.warning(f"识别的模块数量不足4个，将重新尝试")
                # 重新尝试，明确要求至少4个模块
                template += "\n\n请确保至少提供4个模块，每个模块应该有不同的焦点。"
                prompt = PromptTemplate(
                    template=template,
                    input_variables=["topic", "key_info"]
                )
                chain = prompt | self.llm
                modules_text = chain.invoke({"topic": topic, "key_info": key_info})
                
                # 重新解析
                modules = []
                for line in modules_text.split('\n'):
                    match = re.match(r'^\s*\d+[\.\、]\s*(.*?)$', line)
                    if match:
                        modules.append(match.group(1).strip())
            
            # 如果模块数量超过8个，只保留前8个
            if len(modules) > 8:
                logger.warning(f"识别的模块数量超过8个，将只保留前8个")
                modules = modules[:8]
            
            logger.info(f"已识别{len(modules)}个模块: {', '.join(modules)}")
            return modules
        except Exception as e:
            logger.error(f"识别模块失败: {e}")
            # 返回默认模块
            default_modules = [
                f"{topic}的背景与概述",
                f"{topic}的主要特点",
                f"{topic}的发展趋势",
                f"{topic}的应用案例"
            ]
            logger.info(f"使用默认模块: {', '.join(default_modules)}")
            return default_modules
    
    def _batch_allocate_content(self, key_info: str, modules: List[str], topic: str) -> Dict[str, str]:
        """分批处理长文本内容分配"""
        logger.info("分批处理内容分配")
        
        # 将关键信息分成较小的批次
        info_chunks = self.text_splitter.split_text(key_info)
        logger.info(f"关键信息已分为{len(info_chunks)}个块进行处理")
        
        # 初始化每个模块的内容
        result = {}
        
        # 对每个模块单独处理
        for module in modules:
            logger.info(f"为模块 '{module}' 提取相关内容")
            module_content = []
            
            # 对每个信息块提取与当前模块相关的内容
            for i, chunk in enumerate(info_chunks):
                logger.info(f"处理模块 '{module}' 的内容块 {i+1}/{len(info_chunks)}")
                
                template = """
                请从以下关键信息中，提取与"{module}"相关的内容。
                
                主题: {topic}
                模块: {module}
                
                关键信息:
                {chunk}
                
                请仅提取与该模块直接相关的内容，确保:
                1. 只提取与模块主题高度相关的内容
                2. 保留原文中的关键数据和事实
                3. 使用简洁明了的语言
                4. 如果没有相关内容，请直接回复"无相关内容"
                
                相关内容:
                """
                
                prompt = PromptTemplate(
                    template=template,
                    input_variables=["module", "topic", "chunk"]
                )
                
                try:
                    chain = prompt | self.llm
                    content = chain.invoke({
                        "module": module,
                        "topic": topic,
                        "chunk": chunk
                    })
                    
                    # 如果有相关内容，添加到模块内容列表
                    if content and "无相关内容" not in content:
                        module_content.append(content)
                    
                    # 避免API限流
                    time.sleep(1)
                except Exception as e:
                    logger.error(f"处理模块 '{module}' 的内容块 {i+1} 失败: {e}")
            
            # 合并该模块的所有内容
            if module_content:
                combined_content = "\n\n".join(module_content)
                
                # 如果内容较长，进行整合优化
                if len(combined_content) > 5000:
                    logger.info(f"优化模块 '{module}' 的内容")
                    
                    optimize_template = """
                    请对以下关于"{module}"的内容进行整合和优化。
                    
                    主题: {topic}
                    
                    原始内容:
                    {content}
                    
                    请整合内容，确保:
                    1. 去除重复的信息
                    2. 保留所有重要的事实和数据
                    3. 内容逻辑连贯，层次分明
                    4. 使用简洁明了的语言
                    
                    整合后的内容:
                    """
                    
                    optimize_prompt = PromptTemplate(
                        template=optimize_template,
                        input_variables=["module", "topic", "content"]
                    )
                    
                    try:
                        chain = optimize_prompt | self.llm
                        optimized_content = chain.invoke({
                            "module": module,
                            "topic": topic,
                            "content": combined_content
                        })
                        result[module] = optimized_content
                    except Exception as e:
                        logger.error(f"优化模块 '{module}' 内容失败: {e}")
                        result[module] = combined_content
                else:
                    result[module] = combined_content
            else:
                logger.warning(f"模块 '{module}' 没有找到相关内容")
                result[module] = ""
        
        logger.info("分批内容分配完成")
        return result
    
    def _allocate_content_to_modules(self, key_info: str, modules: List[str], topic: str) -> Dict[str, str]:
        """将关键信息分配到各个模块"""
        logger.info("将内容分配到各模块")
        
        template = """
        请根据以下关键信息，将内容分配到各个模块中。
        
        主题: {topic}
        
        模块:
        {modules}
        
        关键信息:
        {key_info}
        
        请为每个模块分配相关的内容，确保内容与模块主题相关。允许一定程度的内容重复。
        
        输出格式:
        
        模块: 模块一标题
        内容:
        - 相关内容1
        - 相关内容2
        
        模块: 模块二标题
        内容:
        - 相关内容1
        - 相关内容2
        
        ...
        
        内容分配:
        """
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["topic", "modules", "key_info"]
        )
        
        try:
            # 如果关键信息太长，需要分批处理
            if len(key_info) > 8000:
                logger.info("关键信息过长，将分批处理")
                return self._batch_allocate_content(key_info, modules, topic)
            
            # 格式化模块列表，待调整，分模块进行
            modules_text = "\n".join([f"{i+1}. {module}" for i, module in enumerate(modules)])
            
            chain = prompt | self.llm
            allocation_text = chain.invoke({
                "topic": topic, 
                "modules": modules_text, 
                "key_info": key_info
            })
            
            # 解析分配结果
            allocation = {}
            current_module = None
            current_content = []
            
            for line in allocation_text.split('\n'):
                # 检查是否是模块标题行
                module_match = re.match(r'^模块[:：]?\s*(.*?)$', line)
                if module_match:
                    # 如果已有模块，保存之前的内容
                    if current_module and current_content:
                        allocation[current_module] = '\n'.join(current_content)
                    
                    # 开始新模块
                    current_module = module_match.group(1).strip()
                    current_content = []
                    continue
                
                # 检查是否是内容行
                content_match = re.match(r'^内容[:：]?\s*$', line)
                if content_match:
                    continue
                
                # 如果有当前模块，添加内容
                if current_module and line.strip():
                    current_content.append(line.strip())
            
            # 保存最后一个模块的内容
            if current_module and current_content:
                allocation[current_module] = '\n'.join(current_content)
            
            # 确保所有模块都有内容
            for module in modules:
                if module not in allocation:
                    logger.warning(f"模块 '{module}' 没有分配到内容，将使用空内容")
                    allocation[module] = ""
            
            logger.info(f"内容分配完成，共{len(allocation)}个模块")
            return allocation
        except Exception as e:
            logger.error(f"分配内容失败: {e}")
            # 返回空分配
            return {module: "" for module in modules}
    
    def _summarize_module(self, module_title: str, module_content: str, topic: str) -> str:
        """总结归纳模块内容"""
        logger.info(f"总结模块: {module_title}")
        
        # 如果模块内容为空，返回默认内容
        if not module_content.strip():
            logger.warning(f"模块 '{module_title}' 内容为空，将使用默认内容")
            return f"关于{module_title}的内容不足，需要更多相关资料。"
        
        # 如果内容较短，直接总结
        if len(module_content) < 5000:
            return self._direct_summarize_module(module_title, module_content, topic)
        
        # 如果内容较长，分步总结
        return self._step_summarize_module(module_title, module_content, topic)
    
    def _direct_summarize_module(self, module_title: str, module_content: str, topic: str) -> str:
        """直接总结模块内容"""
        logger.info(f"直接总结模块: {module_title}")
        
        template = """
        请根据以下内容，对"{module_title}"进行总结归纳。
        
        主题: {topic}
        
        内容:
        {module_content}
        
        请进行全面、客观的总结，确保:
        1. 保留原文中的关键数据和事实
        2. 总结内容在1000-2000字之间
        3. 内容逻辑清晰，层次分明
        4. 使用客观、专业的语言
        5. 如有必要，可以使用小标题划分内容
        
        总结:
        """
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["module_title", "topic", "module_content"]
        )
        
        try:
            chain = prompt | self.llm
            summary = chain.invoke({
                "module_title": module_title, 
                "topic": topic, 
                "module_content": module_content
            })
            
            # 检查总结长度
            if len(summary) < self.min_module_content_length:
                logger.warning(f"模块 '{module_title}' 总结过短 ({len(summary)} 字符)，将尝试扩展")
                
                expand_template = """
                请基于以下总结，进行扩展，使内容更加丰富详实。
                
                主题: {topic}
                模块: {module_title}
                
                原总结:
                {summary}
                
                请扩展总结，确保:
                1. 内容在1000-2000字之间
                2. 增加更多细节和例子
                3. 保持内容的准确性和客观性
                
                扩展后的总结:
                """
                
                expand_prompt = PromptTemplate(
                    template=expand_template,
                    input_variables=["module_title", "topic", "summary"]
                )
                
                chain = expand_prompt | self.llm
                summary = chain.invoke({
                    "module_title": module_title, 
                    "topic": topic, 
                    "summary": summary
                })
            
            # 如果总结过长，进行精简
            if len(summary) > self.max_module_content_length:
                logger.warning(f"模块 '{module_title}' 总结过长 ({len(summary)} 字符)，将进行精简")
                
                condense_template = """
                请对以下总结进行精简，使其更加简洁明了。
                
                主题: {topic}
                模块: {module_title}
                
                原总结:
                {summary}
                
                请精简总结，确保:
                1. 内容在1000-2000字之间
                2. 保留最重要的信息和观点
                3. 不丢失关键数据和事实
                
                精简后的总结:
                """
                
                condense_prompt = PromptTemplate(
                    template=condense_template,
                    input_variables=["module_title", "topic", "summary"]
                )
                
                chain = condense_prompt | self.llm
                summary = chain.invoke({
                    "module_title": module_title, 
                    "topic": topic, 
                    "summary": summary
                })
            
            logger.info(f"模块 '{module_title}' 总结完成，长度: {len(summary)} 字符")
            return summary
        except Exception as e:
            logger.error(f"总结模块 '{module_title}' 失败: {e}")
            return f"由于处理错误，无法生成关于{module_title}的总结。"
    
    def _step_summarize_module(self, module_title: str, module_content: str, topic: str) -> str:
        """分步总结长模块内容"""
        logger.info(f"分步总结长模块: {module_title}")
        
        # 将内容分成较小的块
        content_chunks = self.text_splitter.split_text(module_content)
        logger.info(f"模块内容已分为{len(content_chunks)}个块进行处理")
        
        # 第一步：总结每个块
        chunk_summaries = []
        
        for i, chunk in enumerate(content_chunks):
            logger.info(f"总结块 {i+1}/{len(content_chunks)}")
            
            template = """
            请总结以下内容中与"{module_title}"相关的要点。
            
            内容:
            {chunk}
            
            请提取关键信息，确保:
            1. 保留原文中的数据和事实
            2. 使用简洁明了的语言
            3. 总结长度控制在500字以内
            
            总结:
            """
            
            prompt = PromptTemplate(
                template=template,
                input_variables=["module_title", "chunk"]
            )
            
            try:
                chain = prompt | self.llm
                summary = chain.invoke({
                    "module_title": module_title, 
                    "chunk": chunk
                })
                chunk_summaries.append(summary)
                # 避免API限流
                time.sleep(1)
            except Exception as e:
                logger.error(f"总结块 {i+1} 失败: {e}")
                # 如果失败，添加一个空字符串，保持块数量一致
                chunk_summaries.append("")
        
        # 第二步：合并所有块的总结
        combined_summary = "\n\n".join(chunk_summaries)
        
        # 第三步：对合并后的总结进行最终总结
        template = """
        请根据以下内容，对"{module_title}"进行最终总结归纳。
        
        主题: {topic}
        
        内容:
        {combined_summary}
        
        请进行全面、客观的总结，确保:
        1. 保留原文中的关键数据和事实
        2. 总结内容在1000-2000字之间
        3. 内容逻辑清晰，层次分明
        4. 使用客观、专业的语言
        5. 如有必要，可以使用小标题划分内容
        
        最终总结:
        """
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["module_title", "topic", "combined_summary"]
        )
        
        try:
            chain = prompt | self.llm
            final_summary = chain.invoke({
                "module_title": module_title, 
                "topic": topic, 
                "combined_summary": combined_summary
            })
            
            # 检查总结长度
            if len(final_summary) < self.min_module_content_length:
                logger.warning(f"模块 '{module_title}' 最终总结过短 ({len(final_summary)} 字符)，将尝试扩展")
                
                expand_template = """
                请基于以下总结，进行扩展，使内容更加丰富详实。
                
                主题: {topic}
                模块: {module_title}
                
                原总结:
                {final_summary}
                
                请扩展总结，确保:
                1. 内容在1000-2000字之间
                2. 增加更多细节和例子
                3. 保持内容的准确性和客观性
                
                扩展后的总结:
                """
                
                expand_prompt = PromptTemplate(
                    template=expand_template,
                    input_variables=["module_title", "topic", "final_summary"]
                )
                
                chain = expand_prompt | self.llm
                final_summary = chain.invoke({
                    "module_title": module_title, 
                    "topic": topic, 
                    "final_summary": final_summary
                })
            
            logger.info(f"模块 '{module_title}' 最终总结完成，长度: {len(final_summary)} 字符")
            return final_summary
        except Exception as e:
            logger.error(f"分步总结模块 '{module_title}' 失败: {e}")
            return f"由于处理错误，无法生成关于{module_title}的总结。"
    
    def _extract_references(self, module_title: str, module_content: str, articles: List[Dict[str, Any]]) -> List[str]:
        """提取模块内容的参考来源"""
        logger.info(f"提取模块 '{module_title}' 的参考来源")
        
        references = set()
        
        # 从所有文章中提取标题和URL
        article_info = []
        for article in articles:
            title = article.get("title", "")
            url = article.get("url", "")
            if title and url:
                article_info.append({"title": title, "url": url})
        
        # 如果没有文章信息，返回空列表
        if not article_info:
            logger.warning("没有可用的文章信息，无法提取参考来源")
            return []
        
        # 将模块内容分成较小的块
        content_chunks = []
        if len(module_content) > 5000:
            content_chunks = self.text_splitter.split_text(module_content)
        else:
            content_chunks = [module_content]
        
        # 对每个块提取参考来源
        for i, chunk in enumerate(content_chunks):
            logger.info(f"处理参考来源块 {i+1}/{len(content_chunks)}")
            
            template = """
            请根据以下内容，确定其中可能引用的文章来源。
            
            内容:
            {chunk}
            
            可能的文章来源:
            {article_info}
            
            请列出内容中可能引用的文章标题，确保:
            1. 只列出确实被引用的文章
            2. 如果无法确定，请不要猜测
            
            引用的文章标题:
            """
            
            prompt = PromptTemplate(
                template=template,
                input_variables=["chunk", "article_info"]
            )
            
            try:
                # 将文章信息格式化为字符串
                article_info_text = "\n".join([f"{i+1}. 标题: {info['title']}, 来源: {info['url']}" for i, info in enumerate(article_info)])
                
                chain = prompt | self.llm
                refs_text = chain.invoke({
                    "chunk": chunk,
                    "article_info": article_info_text
                })
                
                # 解析引用的文章标题
                for line in refs_text.split('\n'):
                    line = line.strip()
                    if not line:
                        continue
                    
                    # 尝试匹配文章标题
                    for info in article_info:
                        title = info["title"]
                        url = info["url"]
                        
                        # 如果行中包含文章标题，添加到引用集合
                        if title.lower() in line.lower():
                            references.add(f"{title} ({url})")
                
                # 避免API限流
                time.sleep(1)
            except Exception as e:
                logger.error(f"提取参考来源块 {i+1} 失败: {e}")
        
        logger.info(f"模块 '{module_title}' 参考来源提取完成，共{len(references)}个来源")
        return list(references)
    
    def _create_summary(self, topic: str, modules: List[ModuleContent]) -> str:
        """创建总体摘要"""
        logger.info("创建总体摘要")
        
        # 提取所有模块的标题和内容摘要
        modules_info = []
        for module in modules:
            # 提取内容的前200个字符作为摘要
            content_preview = module.content[:200] + "..." if len(module.content) > 200 else module.content
            modules_info.append(f"模块: {module.title}\n摘要: {content_preview}")
        
        modules_text = "\n\n".join(modules_info)
        
        template = """
        请根据以下模块信息，为主题"{topic}"创建一个总体摘要。
        
        模块信息:
        {modules_text}
        
        请创建一个全面、客观的总体摘要，确保:
        1. 概述主题的核心内容和重要性
        2. 简要提及各个模块的主要内容
        3. 总结长度在500-800字之间
        4. 使用客观、专业的语言
        
        总体摘要:
        """
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["topic", "modules_text"]
        )
        
        try:
            chain = prompt | self.llm
            summary = chain.invoke({
                "topic": topic,
                "modules_text": modules_text
            })
            
            logger.info("总体摘要创建完成")
            return summary
        except Exception as e:
            logger.error(f"创建总体摘要失败: {e}")
            return f"关于{topic}的研究报告，包含{len(modules)}个主要模块。"
    
    def analyse(self, document_path: str, topic: str) -> AnalysisResult:
        """分析文档内容，生成结构化报告"""
        logger.info(f"开始分析文档，主题: {topic}")
        
        # 读取文档
        articles = self._read_document(document_path)
        if not articles:
            logger.error("文档读取失败或为空")
            return AnalysisResult(
                topic=topic,
                summary=f"无法分析主题 '{topic}'，文档读取失败或为空。",
                modules=[]
            )
        
        # 提取关键信息
        key_info = self._extract_key_information(articles, topic)
        
        # 识别模块
        modules = self._identify_modules(key_info, topic)
        
        # 分配内容到模块
        module_contents = self._allocate_content_to_modules(key_info, modules, topic)
        
        # 处理每个模块
        processed_modules = []
        for module_title in modules:
            # 获取模块内容
            module_content = module_contents.get(module_title, "")
            
            # 总结模块内容
            summarized_content = self._summarize_module(module_title, module_content, topic)
            
            # 提取参考来源
            references = self._extract_references(module_title, summarized_content, articles)
            
            # 创建模块内容对象
            module = ModuleContent(
                title=module_title,
                content=summarized_content,
                references=references
            )
            
            processed_modules.append(module)
        
        # 创建总体摘要
        summary = self._create_summary(topic, processed_modules)
        
        # 创建分析结果
        result = AnalysisResult(
            topic=topic,
            summary=summary,
            modules=processed_modules
        )
        
        logger.info(f"文档分析完成，主题: {topic}，共{len(processed_modules)}个模块")
        return result
    
    def save_result(self, result: AnalysisResult, output_path: Optional[str] = None) -> str:
        """保存分析结果到文件"""
        if output_path is None:
            # 创建输出目录
            output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
            os.makedirs(output_dir, exist_ok=True)
            
            # 生成文件名
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{result.topic.replace(' ', '_')}_{timestamp}.md"
            output_path = os.path.join(output_dir, filename)
        
        logger.info(f"保存分析结果到: {output_path}")
        
        # 生成Markdown内容
        content = [
            f"# {result.topic}",
            "",
            "## 摘要",
            "",
            result.summary,
            ""
        ]
        
        # 添加模块内容
        for i, module in enumerate(result.modules):
            content.extend([
                f"## {i+1}. {module.title}",
                "",
                module.content,
                ""
            ])
            
            # 添加参考来源
            if module.references:
                content.extend([
                    "### 参考来源",
                    ""
                ])
                for ref in module.references:
                    content.append(f"- {ref}")
                content.append("")
        
        # 写入文件
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(content))
            logger.info(f"分析结果已保存到: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"保存分析结果失败: {e}")
            return ""


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="内容分析工具")
    parser.add_argument("--document", help="文档路径")
    parser.add_argument("--query", help="分析主题")
    parser.add_argument("--output", "-o", help="输出文件路径")
    parser.add_argument("--api_key", help="通义千问API密钥")
    
    args = parser.parse_args()
    
    # 创建分析Agent
    agent = AnalyseAgent(tongyi_api_key=args.api_key)
    
    # 分析文档
    result = agent.analyse(args.document, args.query)
    
    # 保存结果
    output_path = agent.save_result(result, args.output)
    
    if output_path:
        print(f"分析结果已保存到: {output_path}")
    else:
        print("分析结果保存失败")
                    