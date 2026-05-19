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
done_reasoning = False
for chunk in response:
    if chunk.choices and chunk.choices[0].delta:
        delta = chunk.choices[0].delta
        answer_chunk = delta.content
        if answer_chunk != '':
            if not done_reasoning:
                print('\n\n === Final Answer ===\n')
                done_reasoning = True
            print(answer_chunk, end='', flush=True)