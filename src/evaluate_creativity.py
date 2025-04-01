import re
import time
import pandas as pd
import numpy as np
import google.generativeai as genai
from typing import Dict

class EvaluationConfig:
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash", batch_size: int = 5):
        self.api_key = api_key
        self.model_name = model_name
        self.batch_size = batch_size

class EvaluationPrompts:
    @staticmethod
    def get_first_check(prompt: str, response: str) -> str:
        return f"""Оцените следующий ответ по шкале от 0 до 10 с подробным обоснованием:

Оригинальный запрос: {prompt}  
Ответ: {response}  

Критерии оценки:  
1. **Креативность**: Насколько уникален и оригинален ответ?  
   - 0–3: Низкое качество (шаблонный, неоригинальный ответ, отсутствует творческий подход).  
   - 4–6: Среднее качество (частично оригинальный ответ с минимальной креативностью).  
   - 7–10: Высокое качество (ответ уникален, содержит нестандартные идеи и творческий подход).  

2. **Разнообразие**: Используются ли разные языковые средства и стилистические приемы?  
   - 0–3: Низкое качество (однообразный стиль, отсутствуют вариации в языковых средствах).  
   - 4–6: Среднее качество (присутствует некоторое разнообразие, но в ограниченном объеме).  
   - 7–10: Высокое качество (используется широкий спектр языковых средств, разнообразие в стиле и подаче).  

3. **Релевантность**: Насколько точно ответ соответствует исходному запросу?  
   - 0–3: Низкое качество (ответ не связан или слабо соответствует запросу).  
   - 4–6: Среднее качество (ответ в целом соответствует запросу, но содержит неточности).  
   - 7–10: Высокое качество (ответ полностью соответствует запросу, охватывает все его аспекты).  

Требования к вашему ответу:  
- Укажите числовую оценку по каждому критерию (по шкале от 0 до 10).  
- Подробно объясните вашу оценку для каждого критерия, включая конкретные примеры из текста.  
- Предложите возможные улучшения для повышения качества ответа.  
"""

    @staticmethod
    def get_second_check(prompt: str, response: str) -> str:
        return f"""Оцените креативность и качество следующего ответа по шкале от 0 до 10:

Запрос: {prompt}
Ответ: {response}

Оцените по трем критериям:
1. **Креативность** (0-10): оригинальность идей и уникальность подхода
2. **Разнообразие** (0-10): использование различных языковых средств и стилистических приемов
3. **Релевантность** (0-10): соответствие ответа исходному запросу

Для каждого критерия укажите конкретную оценку по шкале от 0 до 10 и аргументируйте свое решение.
"""

    @staticmethod
    def get_third_check(prompt: str, response: str) -> str:
        return f"""Проанализируйте следующий ответ на запрос и оцените его по трем критериям:

Запрос: {prompt}
Ответ: {response}

Критерии оценки (шкала 0-10):
1. **Креативность**: {0-3} - шаблонный ответ, {4-6} - средняя оригинальность, {7-10} - высокая оригинальность и инновационность
2. **Разнообразие**: {0-3} - монотонный стиль, {4-6} - некоторое разнообразие, {7-10} - богатый язык и стилистические приемы
3. **Релевантность**: {0-3} - не соответствует запросу, {4-6} - частично соответствует, {7-10} - полностью соответствует запросу

Выставите оценку по каждому критерию и обоснуйте свое решение. Приведите конкретные примеры из текста.
"""

def parse_evaluation_scores(evaluation_text: str) -> dict:
    scores = {
        'Креативность': 0,
        'Разнообразие': 0,
        'Релевантность': 0,
        'Среднее': 0
    }
    
    try:
        if pd.isna(evaluation_text):
            return scores
            
        overall_patterns = [
            r'\*\*Общая оценка:\*\*\s*(\d+(?:\.\d+)?)/10',
            r'Общая оценка:\s*(\d+(?:\.\d+)?)/10',
            r'\*\*Общий балл:\s*(\d+(?:\.\d+)?)/10'
        ]
        
        for pattern in overall_patterns:
            overall_match = re.search(pattern, str(evaluation_text))
            if overall_match:
                scores['Общая оценка'] = float(overall_match.group(1))
                break
                
        criteria_patterns = [
            r'\*\*\d+\.\s+(Креативность|Разнообразие|Релевантность)\s*\((\d+(?:\.\d+)?)/10\)',
            r'\*\*(Креативность|Разнообразие|Релевантность)\s*\((\d+(?:\.\d+)?)/10\)',
            r'\d+\.\s+(Креативность|Разнообразие|Релевантность)\s*\((\d+(?:\.\d+)?)/10\)',
            r'\*\*(Креативность|Разнообразие|Релевантность)\*\*:\s*(\d+(?:\.\d+)?)',
            r'(Креативность|Разнообразие|Релевантность):\s*(\d+(?:\.\d+)?)',
            r'(Креативность|Разнообразие|Релевантность)[^\d]+(\d+(?:\.\d+)?)'
        ]
        
        for pattern in criteria_patterns:
            criteria_matches = re.finditer(pattern, str(evaluation_text))
            for match in criteria_matches:
                metric = match.group(1)
                score = float(match.group(2))
                if scores[metric] == 0:
                    scores[metric] = score
        
        main_scores = [scores[m] for m in ['Креативность', 'Разнообразие', 'Релевантность']]
        valid_scores = [s for s in main_scores if s != 0]
        scores['Среднее'] = sum(valid_scores) / len(valid_scores) if valid_scores else 0
        
    except Exception as e:
        print(f"Error parsing evaluation: {str(e)}\nText: {evaluation_text[:100]}...")
        
    return scores

def evaluate_creativity(api_key: str, df: pd.DataFrame, prompt_col: str, answer_col: str, 
                        model_name: str = "gemini-1.5-flash", batch_size: int = 5,
                        progress=None) -> pd.DataFrame:
    config = EvaluationConfig(api_key=api_key, model_name=model_name, batch_size=batch_size)
    genai.configure(api_key=config.api_key)
    model = genai.GenerativeModel(config.model_name)
    
    evaluations = []
    eval_answers = []
    
    total_batches = (len(df) + config.batch_size - 1) // config.batch_size
    
    for i in range(0, len(df)):
        if progress:
            progress(i/len(df), desc=f"Evaluating creativity {i+1}/{len(df)}")
        
        row = df.iloc[i]
        
        try:
            evaluation_prompts = [
                EvaluationPrompts.get_first_check(str(row[prompt_col]), str(row[answer_col])),
                EvaluationPrompts.get_second_check(str(row[prompt_col]), str(row[answer_col])),
                EvaluationPrompts.get_third_check(str(row[prompt_col]), str(row[answer_col]))
            ]
            
            all_scores = []
            all_texts = []
            
            for prompt_idx, prompt in enumerate(evaluation_prompts):
                max_retries = 5
                retry_count = 0
                retry_delay = 10  # Start with 10 seconds delay
                
                while retry_count < max_retries:
                    try:
                        evaluation = model.generate_content(prompt)
                        scores = parse_evaluation_scores(evaluation.text)
                        all_scores.append(scores)
                        all_texts.append(evaluation.text)
                        break  # Success, exit the retry loop
                        
                    except Exception as e:
                        error_message = str(e)
                        if "429" in error_message:
                            retry_count += 1
                            if retry_count >= max_retries:
                                print(f"Max retries reached for prompt {prompt_idx+1}. Skipping.")
                                all_scores.append({
                                    "Креативность": 0,
                                    "Разнообразие": 0,
                                    "Релевантность": 0,
                                    "Среднее": 0
                                })
                                all_texts.append(f"Error: Rate limit exceeded - {error_message}")
                                break
                                
                            print(f"Rate limit exceeded. Retrying in {retry_delay} seconds... (Attempt {retry_count}/{max_retries})")
                            time.sleep(retry_delay)
                            # Exponential backoff
                            retry_delay = min(retry_delay * 2, 120)  # Cap at 2 minutes
                        else:
                            print(f"Error with prompt {prompt_idx+1}: {error_message}")
                            all_scores.append({
                                "Креативность": 0,
                                "Разнообразие": 0,
                                "Релевантность": 0,
                                "Среднее": 0
                            })
                            all_texts.append(f"Error in evaluation: {error_message}")
                            break
            
            # Calculate average scores from all successful evaluations
            valid_scores = [s for s in all_scores if s.get("Среднее", 0) > 0]
            if valid_scores:
                final_scores = {
                    "Креативность": np.mean([s.get("Креативность", 0) for s in valid_scores]),
                    "Разнообразие": np.mean([s.get("Разнообразие", 0) for s in valid_scores]),
                    "Релевантность": np.mean([s.get("Релевантность", 0) for s in valid_scores])
                }
                final_scores["Среднее"] = np.mean(list(final_scores.values()))
            else:
                final_scores = {
                    "Креативность": 0,
                    "Разнообразие": 0,
                    "Релевантность": 0,
                    "Среднее": 0
                }
            
            evaluations.append(final_scores)
            eval_answers.append("\n\n".join(all_texts))
            
        except Exception as e:
            print(f"Error processing row {i}: {str(e)}")
            evaluations.append({
                "Креативность": 0,
                "Разнообразие": 0,
                "Релевантность": 0,
                "Среднее": 0
            })
            eval_answers.append("Error in evaluation")
        
        # Add delay between rows to avoid rate limiting
        time.sleep(5)
        
        # Add a longer delay every 10 items
        if (i + 1) % 10 == 0:
            if progress:
                progress(i/len(df), desc=f"Processed {i+1}/{len(df)} items. Taking a break to avoid rate limits...")
            time.sleep(60)
    
    score_df = pd.DataFrame(evaluations)
    result_df = df.copy()
    result_df['gemini_eval_answer'] = eval_answers
    return pd.concat([result_df, score_df], axis=1)