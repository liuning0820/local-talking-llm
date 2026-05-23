from openai import OpenAI

client = OpenAI(
    base_url='http://127.0.0.1:8080/v1',
    api_key='sk-no-key-required', # llama.cpp server doesn't require a real API key
)

response = client.chat.completions.create(
    model='DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf', # model name (ignored by llama.cpp server, must be empty string)
    messages=[
        {
            'role': 'user',
            'content': '1+2=?'
        }
    ],
    stream=True
)
started_reasoning = False
started_answer = False
for chunk in response:
    if chunk.choices and chunk.choices[0].delta:
        delta = chunk.choices[0].delta

        reasoning_chunk = getattr(delta, 'reasoning_content', None)
        if reasoning_chunk:
            if not started_reasoning:
                print('\n\n === Reasoning ===\n')
                started_reasoning = True
            print(reasoning_chunk, end='', flush=True)

        answer_chunk = delta.content
        if answer_chunk:
            if not started_answer:
                print('\n\n === Final Answer ===\n')
                started_answer = True
            print(answer_chunk, end='', flush=True)
