import json
import os

train_path = r'D:\Doctoral_study\CourseSelection\S4 Courses\DOTE6635 AIforBusiness\Assignments\PaperReplication\LLM_News\Finetune ChatGPT\train.jsonl'

print('=== Training Data Verification ===')
print(f'File: {train_path}')
print(f'Exists: {os.path.exists(train_path)}')

if os.path.exists(train_path):
    num_examples = 0
    format_errors = []
    sample_data = None

    with open(train_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                example = json.loads(line)
                if num_examples == 0:
                    sample_data = example
                num_examples += 1
            except json.JSONDecodeError as e:
                format_errors.append(f'Line {line_num}: {e}')

    print(f'\nExamples: {num_examples:,}')
    print(f'Paper requirement: ~12,000')
    print(f'Match: YES' if abs(num_examples - 12376) < 1000 else 'NO')

    print(f'\nFormat errors: {len(format_errors)}')
    if len(format_errors) == 0:
        print('Status: All lines valid')

    if sample_data:
        print(f'\nData structure:')
        print(f'  Keys: {list(sample_data.keys())}')
        if 'messages' in sample_data:
            print(f'  Message count: {len(sample_data["messages"])}')
            for i, msg in enumerate(sample_data['messages']):
                role = msg.get('role', 'unknown')
                content_len = len(msg.get('content', ''))
                print(f'    {i+1}. role={role}, content_length={content_len}')

    # Count total tokens
    try:
        import tiktoken
        encoding = tiktoken.get_encoding('cl100k_base')
        total_tokens = 0
        with open(train_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    example = json.loads(line)
                    for msg in example.get('messages', []):
                        total_tokens += len(encoding.encode(msg.get('content', '')))

        print(f'\nTotal tokens: {total_tokens:,}')
        print(f'Estimated cost (gpt-4o-mini): ${total_tokens / 1000000 * 1.50:.2f}')
        print(f'Estimated cost (gpt-4o): ${total_tokens / 1000000 * 25.00:.2f}')
    except ImportError:
        print('\ntiktoken not available - skipping token count')
else:
    print('ERROR: File not found')
