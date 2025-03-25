# 行业调研助手 (Industry Research Assistant)

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-brightgreen)

## 项目简介

行业调研助手是一个基于AI的自动化行业研究工具，旨在帮助研究人员、分析师和决策者快速获取、分析和整合行业信息，生成结构化的研究报告。本项目利用先进的自然语言处理技术，实现了从信息搜集、内容提取到报告生成的全流程自动化。

## 核心功能

- **智能搜索**：自动改写查询，从多个角度获取全面的行业信息
- **内容提取**：智能识别和提取文章中的关键信息
- **模块化分析**：自动将内容分类到不同的研究模块
- **报告生成**：生成结构化、易读的行业研究报告
- **API服务**：提供RESTful API接口，方便集成到其他系统

## 技术架构

- 基于Python的模块化设计
- 使用大型语言模型进行内容理解和生成
- 采用Langchain框架构建智能代理
- FastAPI提供API服务

## 安装指南

### 环境要求

- Python 3.8+
- 依赖包见requirements.txt

### 安装步骤

```bash
# 克隆仓库
git clone https://github.com/yourusername/research_agent.git
cd research_agent

# 安装依赖
pip install -r requirements.txt

# 设置环境变量
export DASHSCOPE_API_KEY="your_dashscope_api_key"  # 通义千问API密钥
export TAVILY_API_KEY="your_tavily_api_key"  # Tavily搜索API密钥