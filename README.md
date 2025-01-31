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
Output: (using seed 42)
```md
Using device: cuda:0
new rope theta 31250.0
input length: [45, 242, 52, 48]
Prompt:
[{'role': 'user', 'content': 'what is the recipe of mayonnaise?'}]
Response:
I don't have a traditional recipe for mayonnaise, as it is a proprietary recipe developed by Louis Pasteur in the mid-19th century. However, I can provide you with a simplified recipe that approximates the original:

Ingredients:

* 1 cup (240 ml) egg yolks
* 1/2 cup (120 ml) neutral-tasting oil, such as canola or grapeseed oil
* 1 tablespoon lemon juice or white vinegar
* Salt, to taste

Instructions:

1. In a small bowl, whisk together the egg yolks and lemon juice or vinegar until well combined.
2. Slowly pour the oil into the bowl while continuously whisking the egg yolks.
3. Continue whisking until the mixture thickens and emulsifies, which should take about 5-7 minutes.
4. Remove the bowl from the heat and stir in the salt.
5. Let the mixture cool to room temperature, then refrigerate for at least 30 minutes before serving.

Note: This recipe is a simplified version of the traditional mayonnaise recipe, and the result may not be identical to the original. However, it should still provide a tasty and creamy mayonnaise.

================================================================================
Prompt:
[{'role': 'user', 'content': 'I am going to Paris, what should I see?'}, {'role': 'assistant', 'content': "Paris, the capital of France, is known for its stunning architecture, art museums, historical landmarks, and romantic atmosphere. Here are some of the top attractions to see in Paris:\n\n1. The Eiffel Tower: The iconic Eiffel Tower is one of the most recognizable landmarks in the world and offers breathtaking views of the city.\n2. The Louvre Museum: The Louvre is one of the world's largest and most famous museums, housing an impressive collection of art and artifacts, including the Mona Lisa.\n3. Notre-Dame Cathedral: This beautiful cathedral is one of the most famous landmarks in Paris and is known for its Gothic architecture and stunning stained glass windows.\n\nThese are just a few of the many attractions that Paris has to offer. With so much to see and do, it's no wonder that Paris is one of the most popular tourist destinations in the world."}, {'role': 'user', 'content': 'What is so great about #1?'}]
Response:
The Eiffel Tower is a iconic and awe-inspiring landmark in Paris, and here are some reasons why:

1. **Unmatched Height**: The Eiffel Tower stands at an impressive 324 meters (1,063 feet) tall, making it the tallest structure in Paris.

2. **Engineering Marvel**: The Eiffel Tower was the world's tallest man-made structure when it was first built for the 1889 World's Fair. It has been modified and expanded over the years, but its basic design is still impressive.

3. **Panoramic Views**: The Eiffel Tower offers breathtaking 360-degree views of the city of Paris and the surrounding landscape.

4. **Iconic Symbolism**: The Eiffel Tower is an instantly recognizable symbol of Paris and France, making it a popular tourist destination.

5. **Rich History**: The Eiffel Tower has a rich history, having been built for the 1889 World's Fair and having been used as a symbol of French culture and engineering.

Overall, the Eiffel Tower is an iconic and awe-inspiring landmark that is steeped in history and symbolism.

================================================================================
Prompt:
[{'role': 'system', 'content': 'Always answer with Haiku'}, {'role': 'user', 'content': 'I am going to Paris, what should I see?'}]
Response:
Eiffel Tower gleams
River Seine's gentle flow
Romance abounds

================================================================================
Prompt:
[{'role': 'system', 'content': 'Always answer with emojis'}, {'role': 'user', 'content': 'How to go from Beijing to NY?'}]
Response:
🛫️🗽️ Here's a step-by-step guide to help you travel from Beijing to New York:

**By Air:**

1. **From Beijing Capital International Airport (PEK)**:
        * Take a taxi or a ride-hailing service like Didi Chuxing.
        * Alternatively, you can take a bus from the airport to the city center.
2. **From Beijing's major train stations**:
        * Take a train from Beijing's major train stations like Beijing West Railway Station (PKW), Beijing North Railway Station (PKN), or Beijing South Railway Station (PKS) to the city center.
        * You can also take a bus or a taxi from the train station to the city center.

**By Bus:**

1. **From Beijing's major bus stations**:
        * Take a bus from Beijing's major bus stations like Beijing South Bus Station (SBS), Beijing North Bus Station (NBS), or Beijing West Bus Station (WSB) to the city center.
        * You can also take a bus from the bus station to the city center.

**By Taxi:**

1. **From Beijing's major taxi stands**:
        * You can hail a taxi on the street or find one at a taxi stand.
        * You can also use a ride-hailing service like Didi Chuxing.

**By Car:**

1. **From Beijing's major roads**:
        * You can drive to Beijing from most cities in China.
        * Make sure to check the road conditions before you leave.

Remember to always follow local traffic laws and regulations, and to be mindful of your surroundings while driving.

================================================================================
```

### References
- https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
- https://github.com/meta-llama/llama3/tree/main
- https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
- https://github.com/rasbt/LLMs-from-scratch/tree/main/ch05/07_gpt_to_llama