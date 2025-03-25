# -*- coding: utf-8 -*-

import argparse
import sys
import os

def main():
    """行业调研助手主入口"""
    parser = argparse.ArgumentParser(description="行业调研助手")
    parser.add_argument("--api", action="store_true", help="启动API服务")
    parser.add_argument("--port", type=int, default=8000, help="API服务端口，默认为8000")
    parser.add_argument("--query", help="直接执行调研查询")
    parser.add_argument("--sources", type=int, default=5, help="需要爬取的来源数量，默认为5")
    
    args = parser.parse_args()
    
    if args.api:
        # 启动API服务
        from api.api_manager import app
        import uvicorn
        print(f"正在启动行业调研助手API服务，端口: {args.port}")
        uvicorn.run(app, host="0.0.0.0", port=args.port)
    elif args.query:
        # 直接执行调研
        # from research_tools.cli import main as cli_main
        sys.argv = [sys.argv[0]] + [args.query, "--sources", str(args.sources)]
        # cli_main()
    else:
        # 显示帮助信息
        parser.print_help()

if __name__ == "__main__":
    main()