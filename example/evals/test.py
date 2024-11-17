from example.config import ai, ls


@ls.task()
async def shit():
    result = await ai.generate_text(
        model="openai:o1-preview",
        prompt="What is the capital of France?",
    )
    print(result, ls._get_current_run().inference_cost)
