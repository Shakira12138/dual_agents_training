import json

# 文件路径
input_file = '/work/projects/polyullm/caishuo/workspace/data/polaris-data-53K-indexed_c1c2c3.jsonl'
output_file = 'polaris.jsonl'

# 开头和结尾要加的内容
header = "Solve the following math problem step by step. The last line of your response should be of the form Answer:  \\boxed{$Answer}  where $Answer is the answer to the problem.\n\n"
footer = "\n\nRemember to put your answer on its own line after \"Answer:\"."

with open(input_file, 'r', encoding='utf-8') as infile, \
     open(output_file, 'w', encoding='utf-8') as outfile:

    for line in infile:
        data = json.loads(line.strip())

        if 'prompt' in data and isinstance(data['prompt'], list) and len(data['prompt']) > 0:
            if 'content' in data['prompt'][0]:
                original_content = data['prompt'][0]['content']
                data['prompt'][0]['content'] = header + original_content + footer

        outfile.write(json.dumps(data, ensure_ascii=False) + '\n')

print("处理完成，新文件已保存为:", output_file)
