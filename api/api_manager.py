from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import logging
import os
import sys
from datetime import datetime

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # 设置日志级别
handler = logging.StreamHandler(sys.stdout)  # 输出到控制台
# 添加日志格式化配置
handler.setFormatter(logging.Formatter(
    '%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d - %(message)s'
))
logger.addHandler(handler)


# 创建FastAPI应用
app = FastAPI(
    title="行业调研助手API",
    description="提供行业调研相关的API服务",
    version="0.0.1"
)


from fastapi.middleware.cors import CORSMiddleware

# 添加CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 添加SearchAgent集成
from agent.search_agent import SearchAgent

# 创建Agent管理器实例
agent_manager = None
search_agent = None

@app.on_event("startup")
async def startup_event():
    """应用启动时执行"""
    global search_agent
    try:
        search_agent = SearchAgent()
        logger.info("API服务已启动，SearchAgent初始化成功")
    except Exception as e:
        logger.error(f"SearchAgent初始化失败: {str(e)}")
        raise

@app.get("/health")
async def health_check():
    """健康检查接口"""
    try:
        logger.info("API服务已启动，Agent管理器初始化成功")
        print("API服务已启动，Agent管理器初始化成功")
    except Exception as e:
        logger.error(f"Agent管理器初始化失败: {str(e)}")
        raise
    return {"status": "ok"}

# 定义请求模型
class ResearchRequest(BaseModel):
    query: str
    num_rewrites: Optional[int] = 3

# 定义响应模型
class ResearchResponse(BaseModel):
    query: str
    results: List[Dict[str, Any]]



@app.get("/research/{query}")
async def get_query(query: str):
    """
    进行行业调研的接口
    """
    global agent_manager
    try:
        logger.info("开始调研")
        logger.info(f"调研内容：{query}")  # 使用f-string格式化日志
        print("调研开始")
        logger.info(f"调研内容：{query}")

    except Exception as e:
        logger.error(f"Agent管理器初始化失败: {str(e)}")
        raise
    # 调用AgentManager的get_query方法进行调研
    result = f"调研结果{query}"
    logger.info(f"调研结束,结果为：{query}")
    return  {"research": result}


@app.get("/")
async def root():
    """API根路径，返回基本信息"""
    return {
        "name": "行业调研助手API",
        "version": "1.0.0",
        "description": "提供行业调研相关的API服务",
        "docs_url": "/docs",
        "endpoints": [
            {"path": "/health", "method": "GET", "description": "健康检查接口"},
            {"path": "/research/{query}", "method": "GET", "description": "进行调研"},
            # {"path": "/results", "method": "GET", "description": "列出所有已保存的调研结果"},
            # {"path": "/results/{filename}", "method": "GET", "description": "获取特定结果文件的内容"}
        ]
    }

# app = FastAPI(...)


from fastapi.responses import RedirectResponse
@app.get("/", include_in_schema=False)
async def redirect_to_docs():
    return RedirectResponse(url="/docs")
    
if __name__ == "__main__":
    import uvicorn
    # uvicorn.run(app, host="0.0.0.0", port=8000)
    logger.info("Starting server on port 8080")  # 新增启动日志
    # config = uvicorn.Config("api_manager:app", port=8080, log_level="info",reload=True)
    # server = uvicorn.Server(config)
    # server.run()
    uvicorn.run(
        "api_manager:app",
        host="0.0.0.0",
        port=8080,
        access_log=True,  # 启用访问日志
        log_level="info",  # 设置日志级别
        reload=True,  # 启用自动重载
    )