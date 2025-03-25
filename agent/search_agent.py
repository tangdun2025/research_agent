import os
import sys
import logging
from typing import List, Dict, Any, Optional
import tempfile
import requests

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_react_agent
# 修正导入路径
from langchain_community.tools import TavilySearchResults
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
# 导入通义千问集成
from langchain_community.llms import Tongyi

# 配置日志
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter(
    '%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d - %(message)s'
))
logger.addHandler(handler)

# 定义输出模型
class QueryRewrite(BaseModel):
    queries: List[str] = Field(description="改写后的查询列表")

class ArticleScore(BaseModel):
    title: str = Field(description="文章标题")
    url: str = Field(description="文章链接")
    snippet: str = Field(description="文章摘要")
    score: float = Field(description="相关性评分(0-10)")

class ContentChunk(BaseModel):
    title: str = Field(description="文章标题")
    url: str = Field(description="文章链接")
    chunks: List[str] = Field(description="文章内容块")

class SearchAgent:
    """搜索Agent，用于行业调研"""
    
    def __init__(self, api_key: Optional[str] = None, dashscope_api_key: Optional[str] = None):
        """初始化搜索Agent"""
        # 设置API密钥
        if api_key:
            os.environ["TAVILY_API_KEY"] = api_key
            
        dashscope_api_key='xxxxx'
        # 设置通义千问API密钥
        if dashscope_api_key:
            os.environ["DASHSCOPE_API_KEY"] = dashscope_api_key
        
        # 初始化LLM - 使用通义千问
        self.llm = Tongyi(
            model_name="qwen-turbo",
            temperature=0.1,
            #dashscope_api_key=os.environ.get("DASHSCOPE_API_KEY")
        )
        
        # 初始化工具
        self.search_tool = TavilySearchResults(max_results=10)
        
        # 创建Agent
        self._create_agent()
        
    def _create_agent(self):
        """创建Agent"""
        tools = [
            Tool(
                name="Search",
                func=self.search_tool.invoke,
                description="搜索引擎，用于查询网络信息"
            ),
            Tool(
                name="WebReader",
                func=self._read_webpage,
                description="网页阅读器，用于获取网页内容"
            )
        ]
        
        prompt = PromptTemplate.from_template(
            """你是一个专业的行业调研助手。
            
            你需要完成以下任务：
            1. 对用户的查询进行改写，生成多个不同角度的查询
            2. 使用搜索工具获取相关信息
            3. 评估搜索结果的相关性，筛选出最相关的文章
            4. 获取并分析文章内容
            
            你有以下工具可以使用：
            {tools}
            
            工具名称: {tool_names}
            
            请按照以下步骤思考：
            1. 分析用户查询的核心意图
            2. 从不同角度改写查询
            3. 搜索并评估结果
            4. 提取关键信息
            
            用户查询: {query}
            
            {agent_scratchpad}
            """
        )
        
        self.agent = create_react_agent(self.llm, tools, prompt)
        self.agent_executor = AgentExecutor(agent=self.agent, tools=tools, verbose=True)
    
    def rewrite_query(self, query: str, num_rewrites: int = 3) -> List[str]:
        """改写用户查询"""
        logger.info(f"改写查询: {query}")
        
        template = """
        请将以下查询改写为{num_rewrites}个不同角度的查询，以获取更全面的搜索结果。
        原始查询: {query}
        
        请以JSON格式输出，格式为：
        {{
            "queries": [
                "改写查询1",
                "改写查询2",
                "改写查询3"
            ]
        }}
        
        只返回JSON格式，不要有其他文字。
        """
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["query", "num_rewrites"],
            partial_variables={"num_rewrites": num_rewrites}
        )
        
        # 使用LLM生成改写查询
        try:
            # 尝试使用解析器链
            parser = PydanticOutputParser(pydantic_object=QueryRewrite)
            chain = prompt | self.llm | parser
            result = chain.invoke({"query": query, "num_rewrites": num_rewrites})
            all_queries = [query] + result.queries
        except Exception as e:
            # 解析失败时的备用方案
            logger.warning(f"解析器链失败: {e}，使用备用方案")
            response = self.llm.invoke(prompt.format(query=query, num_rewrites=num_rewrites))
            
            try:
                # 尝试手动解析JSON
                import json
                import re
                
                # 提取JSON部分
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    data = json.loads(json_str)
                    rewritten_queries = data.get("queries", [])
                else:
                    # 如果没有找到JSON，尝试提取列表项
                    rewritten_queries = re.findall(r'"([^"]+)"', response)
                
                # 如果仍然没有查询，创建一些基本的改写
                if not rewritten_queries:
                    rewritten_queries = [
                        f"{query} 最新信息",
                        f"{query} 分析",
                        f"{query} 评价"
                    ][:num_rewrites]
                
                all_queries = [query] + rewritten_queries
            except Exception as e2:
                logger.error(f"备用解析也失败: {e2}，使用默认改写")
                # 最后的备用方案：使用简单的默认改写
                all_queries = [
                    query,
                    f"{query} 最新信息",
                    f"{query} 分析",
                    f"{query} 评价"
                ][:num_rewrites+1]
        
        logger.info(f"改写结果: {all_queries}")
        return all_queries
    
    def search(self, queries: List[str]) -> List[Dict[str, Any]]:
        """执行搜索"""
        logger.info(f"执行搜索: {queries}")
        
        all_results = []
        for query in queries:
            results = self.search_tool.invoke(query)
            all_results.extend(results)
            # logger.info(f"搜索结果={query}:{results}")
        return all_results
    
    def score_results(self, query: str, results: List[Dict[str, Any]]) -> List[ArticleScore]:
        """评分筛选搜索结果"""
        logger.info(f"评分筛选结果，共{len(results)}条")
        
        # 如果没有结果，直接返回空列表
        if not results:
            logger.warning("没有搜索结果可供评分")
            return []
        
        template = """
        请根据原始查询，对以下搜索结果进行相关性评分(0-10分)。
        原始查询: {query}
        
        搜索结果:
        {results}
        
        请以JSON格式输出评分结果，格式为：
        [
            {{
                "title": "文章标题1",
                "url": "文章链接1",
                "snippet": "文章摘要1",
                "score": 8.5
            }},
            {{
                "title": "文章标题2",
                "url": "文章链接2",
                "snippet": "文章摘要2",
                "score": 7.2
            }}
        ]
        
        只返回JSON格式，不要有其他文字。
        """
        
        # 将结果格式化为文本
        results_text = "\n\n".join([
            f"标题: {r.get('title', 'N/A')}\n链接: {r.get('url', 'N/A')}\n摘要: {r.get('content', 'N/A')}"
            for r in results
        ])
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["query", "results"]
        )
        
        # 使用LLM评分
        try:
            chain = prompt | self.llm
            result = chain.invoke({"query": query, "results": results_text})
            logger.debug(f"LLM评分结果原始输出: {result}")
            
            # 解析结果
            import json
            import re
            
            # 尝试直接解析JSON
            try:
                scored_results = json.loads(result)
            except json.JSONDecodeError:
                # 如果直接解析失败，尝试提取JSON部分
                logger.warning("直接JSON解析失败，尝试提取JSON部分")
                json_match = re.search(r'\[.*\]', result, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    scored_results = json.loads(json_str)
                else:
                    # 如果仍然失败，创建默认评分
                    logger.error("无法提取JSON，使用默认评分")
                    scored_results = []
                    for r in results[:10]:  # 只处理前10个结果
                        scored_results.append({
                            "title": r.get("title", "未知标题"),
                            "url": r.get("url", ""),
                            "snippet": r.get("content", "无摘要"),
                            "score": 5.0  # 默认中等评分
                        })
            
            # 转换为ArticleScore对象
            scored_articles = []
            for item in scored_results:
                try:
                    # 确保所有必要字段都存在
                    if "title" not in item:
                        item["title"] = "未知标题"
                    if "url" not in item:
                        item["url"] = ""
                    if "snippet" not in item:
                        item["snippet"] = item.get("content", "无摘要")
                    if "score" not in item:
                        item["score"] = 5.0
                    
                    # 确保score是浮点数
                    if isinstance(item["score"], str):
                        item["score"] = float(item["score"].replace(",", "."))
                    
                    scored_articles.append(ArticleScore(**item))
                except Exception as e:
                    logger.warning(f"处理评分项失败: {e}, 项: {item}")
            
            # 按评分排序
            scored_articles.sort(key=lambda x: x.score, reverse=True)
            # 取前10个
            top_articles = scored_articles[:10]
            
            logger.info(f"筛选出{len(top_articles)}篇相关文章")
            return top_articles
            
        except Exception as e:
            logger.error(f"评分过程失败: {e}")
            # 出错时返回原始结果的前10项，但不进行评分
            default_articles = []
            for r in results[:10]:
                default_articles.append(ArticleScore(
                    title=r.get("title", "未知标题"),
                    url=r.get("url", ""),
                    snippet=r.get("content", "无摘要"),
                    score=5.0  # 默认中等评分
                ))
            return default_articles
    
    def _clean_content(self, content: str) -> str:
        """清洗内容，去除无意义内容或字符，并将内容统一编码成汉字"""
        logger.info("清洗网页内容并统一编码为汉字")
        
        if not content or content.startswith("读取失败"):
            return content
            
        # 导入正则表达式模块
        import re
        
        # 移除HTML标签
        content = re.sub(r'<[^>]+>', ' ', content)
        
        # 移除多余空白字符
        content = re.sub(r'\s+', ' ', content)
        
        # 移除特殊Unicode字符和控制字符
        content = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', content)
        
        # 移除常见的网页噪声
        noise_patterns = [
            r'cookie[s]?\s+政策',
            r'隐私政策',
            r'版权所有',
            r'Copyright © \d{4}',
            r'All Rights Reserved',
            r'网站地图',
            r'使用条款',
            r'关注我们',
            r'分享到',
            r'点击加载更多',
            r'返回顶部',
            r'广告',
            r'赞\d+',
            r'评论\d+',
            r'阅读\d+',
            r'\d+阅读',
            r'\d+评论',
            r'\d+赞',
            r'JavaScript is disabled',
            r'Please enable JavaScript',
            r'您的浏览器不支持.*?脚本',
            r'您的浏览器版本过低',
            r'联系我们',
            r'关于我们',
            r'加入我们',
            r'招聘信息',
            r'帮助中心',
            r'常见问题',
            r'免责声明',
            r'举报',
            r'反馈',
        ]
        
        for pattern in noise_patterns:
            content = re.sub(pattern, '', content, flags=re.IGNORECASE)
        
        # 移除PDF格式的无效内容
        content = re.sub(r'%PDF-\d+\.\d+.*?endobj', '', content, flags=re.DOTALL)
        content = re.sub(r'<>stream.*?endstream', '', content, flags=re.DOTALL)
        
        # 移除连续的标点符号
        content = re.sub(r'[.。!！?？,，;；:：]{2,}', '.', content)
        
        # 移除过长的数字序列（可能是ID或其他无意义数据）
        content = re.sub(r'\d{10,}', '', content)
        
        # 移除URL
        content = re.sub(r'https?://\S+', '', content)
        
        # 移除邮箱地址
        content = re.sub(r'\S+@\S+\.\S+', '', content)
        
        # 移除多余的空行
        content = re.sub(r'\n\s*\n', '\n\n', content)
        
        # 将英文标点符号转换为中文标点符号
        # punctuation_map = {
        #     '.': '。',
        #     ',': '，',
        #     ':': '：',
        #     ';': '；',
        #     '?': '？',
        #     '!': '！',
        #     '(': '（',
        #     ')': '）',
        #     '[': '【',
        #     ']': '】',
        #     '"': '"',
        #     "'": ''',
        # }
        
        # for en_punct, cn_punct in punctuation_map.items():
        #     content = content.replace(en_punct, cn_punct)
        
        # 尝试将英文单词转换为中文（使用LLM进行转换）
        # 检测是否包含大量英文
        english_ratio = len(re.findall(r'[a-zA-Z]', content)) / (len(content) + 1)
        
        if english_ratio > 0.3:  # 如果英文字符占比超过30%
            try:
                # 使用已初始化的LLM进行翻译
                template = """
                请将以下文本翻译成标准中文，确保所有内容都使用汉字表达：
                1. 将英文单词和短语翻译成对应的中文
                2. 将专业术语转换为中文常用表达
                3. 保持原文的意思和结构
                4. 确保翻译后的文本流畅自然
                
                {text}
                
                只返回翻译结果，不要有其他解释。
                """
                
                # 为了避免超出LLM的输入限制，将内容分段处理
                max_chunk_size = 2000
                translated_chunks = []
                
                for i in range(0, len(content), max_chunk_size):
                    chunk = content[i:i+max_chunk_size]
                    prompt = PromptTemplate(template=template, input_variables=["text"])
                    
                    # 使用self.llm进行翻译
                    chain = prompt | self.llm
                    translated = chain.invoke({"text": chunk})
                    translated_chunks.append(translated)
                
                content = "".join(translated_chunks)
                logger.info("内容已翻译为中文")
            except Exception as e:
                logger.warning(f"翻译内容失败: {e}，保留原始内容")
        
        # 如果清理后内容太少，可能是过度清理，返回原始内容的一部分
        if len(content.strip()) < 100:
            logger.warning("清洗后内容过少，可能过度清理")
            # 返回原始内容的前2000个字符
            return content[:2000]
        
        logger.info("内容清洗和编码转换完成")
        return content.strip()
    
    def _read_webpage(self, url: str) -> str:
        """读取网页内容，支持HTML和PDF格式"""
        logger.info(f"读取内容: {url}")
        
        try:
            # 检查URL是否指向PDF文件
            if url.lower().endswith('.pdf') or 'application/pdf' in url.lower():
                logger.info(f"检测到PDF文件: {url}")
                return self._read_pdf(url)
            else:
                # 使用WebBaseLoader读取普通网页
                loader = WebBaseLoader(url)
                docs = loader.load()
                return docs[0].page_content
        except Exception as e:
            logger.error(f"读取内容失败: {e}")
            return f"读取失败: {str(e)}"
    
    def _read_pdf(self, url: str) -> str:
        """读取PDF文件并转换为纯文本"""
        import tempfile
        import requests
        import os
        from langchain_community.document_loaders import PyPDFLoader
        
        logger.info(f"开始下载和解析PDF: {url}")
        
        try:
            # 创建临时文件
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                temp_path = temp_file.name
                
                # 下载PDF文件
                logger.info(f"下载PDF到临时文件: {temp_path}")
                response = requests.get(url, stream=True, timeout=30)
                response.raise_for_status()  # 确保请求成功
                
                # 写入临时文件
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        temp_file.write(chunk)
                
                temp_file.flush()
            
            # 使用PyPDFLoader解析PDF
            logger.info(f"解析PDF文件: {temp_path}")
            loader = PyPDFLoader(temp_path)
            documents = loader.load()
            
            # 合并所有页面的内容
            text_content = "\n\n".join([doc.page_content for doc in documents])
            
            # 删除临时文件
            try:
                os.unlink(temp_path)
                logger.info(f"已删除临时文件: {temp_path}")
            except Exception as e:
                logger.warning(f"删除临时文件失败: {e}")
            
            logger.info(f"PDF解析完成，内容长度: {len(text_content)} 字符")
            return text_content
            
        except Exception as e:
            logger.error(f"PDF处理失败: {e}")
            return f"PDF处理失败: {str(e)}"
    
    def chunk_content(self, article: ArticleScore) -> ContentChunk:
        """将文章内容分块"""
        logger.info(f"分块文章: {article.title}")
        
        content = self._read_webpage(article.url)
        
        # 清洗内容并统一编码为汉字
        content = self._clean_content(content)

        # 使用文本分割器
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        chunks = text_splitter.split_text(content)
        
        # 限制块数为5-10个
        if len(chunks) < 5:
            # 如果块数少于5个，减小块大小重新分割
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=100,
                length_function=len,
            )
            chunks = text_splitter.split_text(content)
        
        # 如果块数超过10个，只保留前10个
        chunks = chunks[:10]
        
        logger.info(f"文章分块完成，共{len(chunks)}个块")
        
        return ContentChunk(
            title=article.title,
            url=article.url,
            chunks=chunks
        )
    
    def run(self, query: str, num_rewrites: int = 3) -> List[ContentChunk]:
        """运行搜索Agent"""
        logger.info(f"开始调研: {query}")
        
        # 1. 改写查询
        queries = self.rewrite_query(query, num_rewrites)
        
        # 2. 执行搜索
        search_results = self.search(queries)
        
        # 3. 评分筛选
        scored_articles = self.score_results(query, search_results)
        
        # 4. 内容分块
        content_chunks = []
        for article in scored_articles:
            chunked = self.chunk_content(article)
            content_chunks.append(chunked)
        
        logger.info(f"调研完成，共处理{len(content_chunks)}篇文章")
        
        return content_chunks

def main():
    """主函数"""
    import argparse
    import json
    import datetime
    
    parser = argparse.ArgumentParser(description="行业调研搜索Agent")
    parser.add_argument("query", help="调研查询")
    parser.add_argument("--rewrites", type=int, default=3, help="查询改写数量")
    parser.add_argument("--api-key", help="Tavily API密钥")
    parser.add_argument("--dashscope-api-key", help="通义千问 API密钥")
    parser.add_argument("--output", help="输出文件路径", default="")
    
    args = parser.parse_args()
    
    # 修复：确保查询参数不包含"query="前缀
    query = args.query
    if query.startswith("query="):
        query = query[6:]  # 移除"query="前缀
    
    agent = SearchAgent(api_key=args.api_key, dashscope_api_key=args.dashscope_api_key)
    results = agent.run(query, args.rewrites)
    
    print(f"\n===== 调研结果 =====")
    for i, article in enumerate(results, 1):
        print(f"\n{i}. {article.title}")
        print(f"   链接: {article.url}")
        print(f"   内容块数: {len(article.chunks)}")
    
    # 将结果保存为JSON文件
    if results:
        # 准备JSON数据
        json_data = {
            "query": query,
            "timestamp": datetime.datetime.now().isoformat(),
            "articles": [
                {
                    "title": article.title,
                    "url": article.url,
                    "chunks": article.chunks
                }
                for article in results
            ]
        }
        
        # 确定输出文件路径
        if args.output:
            output_path = args.output
        else:
            # 使用查询和时间戳创建默认文件名
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_query = "".join(c if c.isalnum() else "_" for c in query)[:30]  # 限制长度并替换非法字符
            output_path = f"/Users/yueqi/projects/research_agent/results/{safe_query}_{timestamp}.json"
        
        # 确保目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 保存JSON文件
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n调研结果已保存至: {output_path}")
    else:
        print("\n没有找到相关结果，未生成JSON文件")

    

if __name__ == "__main__":
    main()