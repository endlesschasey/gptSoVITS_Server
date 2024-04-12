import streamlit as st
import requests
import asyncio
import aiohttp
import time
import json
import random

API_URL = "http://127.0.0.1:8000"  # 替换为实际的服务器地址和端口

def add_models(model_configs):
    for model_config in model_configs:
        response = requests.post(f"{API_URL}/add_model", json=model_config)
        if response.status_code != 200:
            raise Exception(f"Error adding model: {response.text}")
    return {"message": "Models added successfully"}

async def generate_audio_concurrent(session, model_name, text, text_language):
    data = {
        "model_name": str(model_name),
        "text": str(text),
        "text_language": text_language
    }
    start_time = time.time()
    async with session.post(f"{API_URL}/generate_audio", json=data) as response:
        result = await response.json()
        end_time = time.time()
        elapsed_time = end_time - start_time
        result["elapsed_time"] = elapsed_time
        return result

async def test_concurrent_generation(selected_models, test_text, test_text_language, concurrent_requests):
    start_time = time.time()
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        model_task_mapping = {}
        result_mapping = {}
        for model_name in selected_models:
            model_task_mapping[model_name] = []
            
            for _ in range(concurrent_requests):
                task = asyncio.ensure_future(generate_audio_concurrent(session, model_name, test_text, test_text_language))
                tasks.append(task)
                model_task_mapping[model_name].append(task)
        
        results = await asyncio.gather(*tasks)
        for i, result in enumerate(results):
            result_mapping[id(result)] = i
    
    end_time = time.time()
    total_time = end_time - start_time
    return results, total_time, model_task_mapping, result_mapping

def analyze_results(results, total_time, model_task_mapping, result_mapping):
    model_analysis = {}
    for model_name, tasks in model_task_mapping.items():
        model_results = [results[result_mapping[id(task.result())]] for task in tasks]
        model_elapsed_times = [result["elapsed_time"] for result in model_results]
        model_total_time = sum(model_elapsed_times)
        model_avg_time = model_total_time / len(model_elapsed_times)
        model_analysis[model_name] = {
            "total_time": model_total_time,
            "avg_time": model_avg_time,
            "num_tasks": len(model_elapsed_times)
        }
    
    return {
        "total_time": total_time,
        "model_analysis": model_analysis
    }

def main():
    st.title("Audio Generation Client")

    # 添加模型
    st.header("Add Models")
    model_count = st.number_input("Number of Models", min_value=1, value=1, step=1)
    model_config_str = st.text_area("Model Configuration (JSON)", height=200)
    if st.button("Add Models"):
        try:
            model_config = json.loads(model_config_str)
            model_configs = []
            for i in range(model_count):
                model_config_copy = model_config.copy()
                model_config_copy["model_name"] = f"model{i+1}"
                model_configs.append(model_config_copy)
            result = add_models(model_configs)
            st.success(result["message"])
        except json.JSONDecodeError:
            st.error("Invalid JSON format")
        except requests.exceptions.RequestException as e:
            st.error(f"Error adding models: {str(e)}")

    # 多模型多任务同时并发测试
    st.header("Concurrent Generation Test")
    selected_models = [f"model{i+1}" for i in range(model_count)]
    
    test_text = st.text_input("Test Text", value="这是一个测试文本")
    test_text_language = st.text_input("Test Text Language", value="中文")
    concurrent_requests = st.number_input("Concurrent Requests", min_value=1, value=5, step=1)
    
    if st.button("Start Concurrent Generation Test"):
        with st.spinner(f"Testing with {model_count} model(s)..."):
            results, total_time, model_task_mapping, result_mapping = asyncio.run(test_concurrent_generation(selected_models, test_text, test_text_language, concurrent_requests))
            analysis = analyze_results(results, total_time, model_task_mapping, result_mapping)
        
        # 使用 Markdown 块呈现最终报告
        report = f"""
        ## Concurrent Generation Test Report ({model_count} model(s))
        
        - Total Time: {analysis['total_time']:.2f}s
        """
        st.markdown(report)
        
        for model_name, model_stats in analysis["model_analysis"].items():
            model_report = f"""
            ### Model: {model_name}
            
            - Number of Tasks: {model_stats['num_tasks']}
            - Total Time: {model_stats['total_time']:.2f}s
            - Average Time per Task: {model_stats['avg_time']:.2f}s
            """
            st.markdown(model_report)
            
if __name__ == "__main__":
    main()