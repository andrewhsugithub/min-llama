### My implementation of the llama3.2-1B

- uses tokenizer from llama3.2-1B-instruct from huggingface
- loads weights from llama3.2-1B-instruct from huggingface

### Run the example
```python
copy .env.template .env
python example.py
```
> Note: remember to fill in the `.env` file with the correct values

### References
- https://github.com/meta-llama/llama3/tree/main
- https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
- https://github.com/rasbt/LLMs-from-scratch/tree/main/ch05/07_gpt_to_llama