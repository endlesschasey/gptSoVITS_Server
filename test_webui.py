import streamlit as st
import requests
import asyncio
import aiohttp
import time
import json
import subprocess
import logging

# 配置日志记录器
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
server_processes = []

def start_server(port):
    server_process = subprocess.Popen(["python", "main.py", "--host", "0.0.0.0", "--port", str(port)])
    time.sleep(5)
    logger.info(f"服务器已在端口 {port} 上启动")
    return server_process

def stop_server(server_process):
    server_process.terminate()
    logger.info("服务器已停止")

def init_manager(api_url, max_models):
    response = requests.post(f"{api_url}/init_manager/{max_models}")
    if response.status_code != 200:
        logger.error(f"初始化管理器时出错: {response.text}")
        raise Exception(f"初始化管理器时出错: {response.text}")
    logger.info(f"管理器在 {api_url} 上初始化成功")

async def add_model(api_url, model_configs, session=None):
    if session is None:
        response = requests.post(f"{api_url}/add_model", json=model_configs)
        if response.status_code != 200:
            logger.error(f"添加模型时出错: {response.text}")
            raise Exception(f"添加模型时出错: {response.text}")
        logger.info(f"模型已成功添加到 {api_url}")
    else:
        start_time = time.time()
        try:
            async with session.post(f"{api_url}/add_model", json=model_configs) as response:
                result = await response.json()
                end_time = time.time()
                elapsed_time = end_time - start_time
                result["elapsed_time"] = elapsed_time
                logger.info(f"服务器 {api_url} 切换模型用时 {elapsed_time:.2f}s")
                return result
        except Exception as e:
            logger.info(f"添加模型失败原因: {str(e)}")
            raise Exception(f"服务器 {api_url} 切换模型时出错")

async def generate_audio_concurrent(session, api_url, model_name, text, text_language, max_retries=3):
    data = {
        "model_name": str(model_name),
        "text": str(text),
        "text_language": text_language
    }
    start_time = time.time()
    
    for attempt in range(max_retries):
        try:
            async with session.post(f"{api_url}/generate_audio", json=data) as response:
                result = await response.json()
                end_time = time.time()
                elapsed_time = end_time - start_time
                result["elapsed_time"] = elapsed_time
                logger.info(f"模型 {model_name} 生成音频用时 {elapsed_time:.2f}s")
                return result
        except aiohttp.ClientError as e:
            logger.error(f"模型 {model_name} 生成音频时出错: {str(e)}")
            if attempt < max_retries - 1:
                await asyncio.sleep(1)  # 等待一段时间后重试
    
    logger.error(f"模型 {model_name} 在 {max_retries} 次尝试后仍然无法生成音频")
    return None

async def test_concurrent_generation(api_urls, model_names, test_text, test_text_language, tasks_per_model):
    start_time = time.time()
    
    async with aiohttp.ClientSession() as session:
        tasks = [
            asyncio.ensure_future(generate_audio_concurrent(session, api_url, model_name, test_text, test_text_language))
            for api_url in api_urls
            for model_name in model_names[api_url]
            for _ in range(tasks_per_model)
        ]
        results = await asyncio.gather(*tasks)
    
    end_time = time.time()
    total_time = end_time - start_time
    logger.info(f"并发生成测试完成,用时 {total_time:.2f}s")
    return results, total_time

async def test_model_switching(api_urls, model_configs, test_text, test_text_language, num_iterations):
    # 并发执行切换模型,得出切换模型的时间成本
    start_time = time.time()
    
    async with aiohttp.ClientSession() as session:
        for i in range(1, num_iterations + 1):
            tasks = [
                    asyncio.ensure_future(add_model(api_url, model_configs[i % len(model_configs)], session))
                    for api_url in api_urls
                ]
            await asyncio.gather(*tasks)
            time.sleep(0.5)
    end_time = time.time()
    total_time = end_time - start_time
    avg_time = total_time / num_iterations
    logger.info(f"模型切换测试完成,用时 {total_time:.2f}s")
    logger.info(f"平均每次迭代用时: {avg_time:.2f}s")
    return total_time, avg_time

async def test_model_switching_with_tasks(api_urls, model_configs, test_text, test_text_language, num_iterations, tasks_per_iteration):
    # 并发执行切换模型和运行一定的任务,得出任务运行中切换模型的时间成本
    start_time = time.time()
    
    async with aiohttp.ClientSession() as session:
            for i in range(num_iterations):
                model_config = model_configs[i % len(model_configs)]
                tasks = [
                    asyncio.ensure_future(generate_audio_concurrent(session, api_url, model_config["model_name"], test_text, test_text_language))
                    for _ in range(tasks_per_iteration)
                    for api_url in api_urls
                ]
                await asyncio.gather(*tasks)
                tasks = [
                    asyncio.ensure_future(add_model(api_url, model_configs[i % len(model_configs)], session))
                    for api_url in api_urls
                ]
                await asyncio.gather(*tasks)
    
    end_time = time.time()
    total_time = end_time - start_time
    avg_time = total_time / (num_iterations * tasks_per_iteration)
    logger.info(f"模型切换并执行任务测试完成,用时 {total_time:.2f}s")
    logger.info(f"平均每个任务用时: {avg_time:.2f}s")
    return total_time, avg_time

def analyze_results(results, total_time):
    elapsed_times = [result["elapsed_time"] for result in results if result is not None]
    min_time = min(elapsed_times) if elapsed_times else 0
    max_time = max(elapsed_times) if elapsed_times else 0
    avg_time = sum(elapsed_times) / len(elapsed_times) if elapsed_times else 0
    
    logger.info(f"分析结果:")
    logger.info(f"  总时间: {total_time:.2f}s")
    logger.info(f"  最短时间: {min_time:.2f}s")
    logger.info(f"  最长时间: {max_time:.2f}s")
    logger.info(f"  平均时间: {avg_time:.2f}s")
    logger.info(f"  总任务数: {len(results)}")
    
    return {
        "total_time": total_time,
        "min_time": min_time,
        "max_time": max_time,
        "avg_time": avg_time,
        "num_tasks": len(results)
    }

async def run_concurrent_generation_test(num_servers, max_models, num_models, tasks_per_model, model_config, test_text, test_text_language):
    # 启动服务器
    api_urls = []
    for i in range(num_servers):
        port = 8000 + i
        server_process = start_server(port)
        server_processes.append(server_process)
        api_url = f"http://localhost:{port}"
        api_urls.append(api_url)
    time.sleep(10)
    # 初始化管理器
    for api_url in api_urls:
        init_manager(api_url, max_models)
    
    # 添加模型
    model_names = [f"model{i+1}" for i in range(num_models)]
    model_configs = [
        {
            "model_name": model_name,
            "gpt_path": model_config["gpt_path"],
            "sovits_path": model_config["sovits_path"],
            "ref_wav_path": model_config["ref_wav_path"],
            "prompt_text": model_config["prompt_text"],
            "prompt_language": model_config["prompt_language"]
        }
        for model_name in model_names
    ]
    use_models = {}
    for api_url in api_urls:
        use_models[api_url] = []    
        for i in range(1, max_models + 1):
            model_config = model_configs[i % len(model_configs)]
            await add_model(api_url, model_config)
            use_models[api_url].append(model_config["model_name"])
    # 并发生成测试
    with st.spinner(f"使用 {num_servers} 个服务器,每个服务器 {num_models} 个模型,每个模型 {tasks_per_model} 个任务进行测试..."):
        results, total_time = await test_concurrent_generation(api_urls, use_models, test_text, test_text_language, tasks_per_model)
        analysis = analyze_results(results, total_time)
    
    return analysis

async def run_model_switching_test(num_servers, max_models, model_config, test_text, test_text_language, num_switch_iterations):
    # 启动服务器
    api_urls = []
    for i in range(num_servers):
        port = 8000 + i
        server_process = start_server(port)
        server_processes.append(server_process)
        api_url = f"http://localhost:{port}"
        api_urls.append(api_url)
    time.sleep(15)
    # 初始化管理器
    for api_url in api_urls:
        init_manager(api_url, max_models)
    
    # 切换模型测试
    with st.spinner(f"使用 {num_switch_iterations} 次迭代进行模型切换测试..."):
        model_configs = [
            {
                "model_name": f"model{i+1}",
                "gpt_path": model_config["gpt_path"],
                "sovits_path": model_config["sovits_path"],
                "ref_wav_path": model_config["ref_wav_path"],
                "prompt_text": model_config["prompt_text"],
                "prompt_language": model_config["prompt_language"]
            }
            for i in range(max_models)
        ]
        switch_total_time, switch_avg_time = await test_model_switching(api_urls, model_configs, test_text, test_text_language, num_switch_iterations)
    
    return switch_total_time, switch_avg_time

async def run_model_switching_with_tasks_test(num_servers, max_models, model_config, test_text, test_text_language, num_switch_tasks_iterations, tasks_per_switch_iteration):
    # 启动服务器
    api_urls = []
    for i in range(num_servers):
        port = 8000 + i
        server_process = start_server(port)
        server_processes.append(server_process)
        api_url = f"http://localhost:{port}"
        api_urls.append(api_url)
    time.sleep(15)
    # 初始化管理器
    for api_url in api_urls:
        init_manager(api_url, max_models)
    
    # 切换模型并执行任务测试
    with st.spinner(f"使用 {num_switch_tasks_iterations} 次迭代,每次迭代 {tasks_per_switch_iteration} 个任务进行模型切换并执行任务测试..."):
        model_configs = [
            {
                "model_name": f"model{i+1}",
                "gpt_path": model_config["gpt_path"],
                "sovits_path": model_config["sovits_path"],
                "ref_wav_path": model_config["ref_wav_path"],
                "prompt_text": model_config["prompt_text"],
                "prompt_language": model_config["prompt_language"]
            }
            for i in range(max_models)
        ]
        switch_tasks_total_time, switch_tasks_avg_time = await test_model_switching_with_tasks(api_urls, model_configs, test_text, test_text_language, num_switch_tasks_iterations, tasks_per_switch_iteration)
    
    return switch_tasks_total_time, switch_tasks_avg_time

def main():
    st.title("语音生成测试工作流")
    
    # 配置测试方案
    st.header("测试配置")
    num_servers = st.number_input("服务器数量", min_value=1, value=1, step=1)
    max_models = st.number_input("每个服务器最大模型数", min_value=1, value=3, step=1)
    num_models = st.number_input("每个服务器模型数", min_value=1, value=1, step=1)
    tasks_per_model = st.number_input("每个模型任务数", min_value=1, value=5, step=1)
    
    model_config_str = st.text_area("模型配置 (JSON)", height=200)
    test_text = st.text_input("测试文本", value="这是一个测试文本")
    test_text_language = st.text_input("测试文本语言", value="中文")
    
    num_switch_iterations = st.number_input("模型切换迭代次数", min_value=1, value=100, step=1)
    num_switch_tasks_iterations = st.number_input("模型切换并执行任务迭代次数", min_value=1, value=100, step=1)
    tasks_per_switch_iteration = st.number_input("每次模型切换迭代任务数", min_value=1, value=5, step=1)
    
    test_options = ["并发生成", "模型切换", "模型切换并执行任务"]
    selected_test = st.selectbox("选择测试", test_options)
    
    if st.button("开始测试"):
        try:
            model_config = json.loads(model_config_str)
            
            if selected_test == "并发生成":
                analysis = asyncio.run(run_concurrent_generation_test(
                    num_servers, max_models, num_models, tasks_per_model, model_config, test_text, test_text_language
                ))
                
                # 输出报告
                report = f"""
                ## 测试报告 - 并发生成
                
                - 服务器数量: {num_servers}
                - 每个服务器模型数: {num_models}
                - 每个模型任务数: {tasks_per_model}
                - 测试文本长度: {len(test_text)}
                - 总时间: {analysis['total_time']:.2f}s
                - 最短时间: {analysis['min_time']:.2f}s
                - 最长时间: {analysis['max_time']:.2f}s
                - 平均时间: {analysis['avg_time']:.2f}s
                - 总任务数: {analysis['num_tasks']}
                """
                st.markdown(report)
            
            elif selected_test == "模型切换":
                switch_total_time, switch_avg_time = asyncio.run(run_model_switching_test(
                    num_servers, max_models, model_config, test_text, test_text_language, num_switch_iterations
                ))
                
                # 输出报告
                report = f"""
                ## 测试报告 - 模型切换
                
                - 迭代次数: {num_switch_iterations}
                - 总时间: {switch_total_time:.2f}s 
                - 平均时间: {switch_avg_time:.2f}s 
                """ 
                
                st.markdown(report)
            
            elif selected_test == "模型切换并执行任务":
                switch_tasks_total_time, switch_tasks_avg_time = asyncio.run(run_model_switching_with_tasks_test(
                    num_servers, max_models, model_config, test_text, test_text_language, num_switch_tasks_iterations, tasks_per_switch_iteration
                ))
                
                # 输出报告
                report = f"""
                ## 测试报告 - 模型切换并执行任务
                
                - 迭代次数: {num_switch_tasks_iterations}
                - 每次迭代任务数: {tasks_per_switch_iteration}
                - 总时间: {switch_tasks_total_time:.2f}s
                - 平均每个任务时间: {switch_tasks_avg_time:.2f}s
                """
                st.markdown(report)
        except json.JSONDecodeError as e:
            logger.error(f"无效的 JSON 格式: {str(e)}")
            st.error("无效的 JSON 格式")
        except Exception as e:
            logger.error(f"错误: {str(e)}")
            st.error(f"错误: {str(e)}")
        finally:
            # 停止服务器
            for server_process in server_processes:
                stop_server(server_process)
            
if __name__ == "__main__":
    main()