# My implementation of the [Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)

- uses tokenizer from Llama-3.2-1B-Instruct from huggingface
- loads weights from Llama-3.2-1B-instruct from huggingface

### Setup
```bash
pip install -r requirements.txt
copy .env.template .env
``` 
> Note: Remember to gain access to the model from [Llama-3.2-1B-Instruct huggingface](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct), then generate a token and add it to the .env file

### Run the example
```python
python example.py
```

### References
- https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
- https://github.com/meta-llama/llama3/tree/main
- https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
- https://github.com/rasbt/LLMs-from-scratch/tree/main/ch05/07_gpt_to_llama