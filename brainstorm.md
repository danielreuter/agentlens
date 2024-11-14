# Refactor brainstorm

Our library, AgentLens, gives you a hooks-based API for observing and modifying the behavior of a big computation graph involving many network requests, database queries, LLM calls, etc.

The computation graph will consist of many nested functions decorated with `@ls.task()`. The library handles the execution of each task, passing in context via contextvars for observability, hooking, caching, mocking, etc. 

The hope is to make it very easy to evaluate specific parts of the graph in a controlled environment.

### General information:
1. Tasks are functions decorated with `@ls.task()`, entering them into a unified observability and evaluation ecosystem
2. The `Lens` object exposes context variables to every task in the graph. 
  - You can manually set these however you like using `ls.context` to define the context type, and `ls.provide` to set the value.
  - You can then get these values in a task by passing the context's type to the getter, e.g. `ls[SomeCustomContextType]`.
  - The file-system within the run folder is nested using the task graph, and you can use that to intuitively display the results at each level, e.g. `ls / "response.md".write_text("...")` will create a file in the run-specific directory, nested all the way under where the task is, containing that response
3. Hooks, Mocks, Evaluations

## Overall task execution

When a task is called, it will check to see whether it is already inside of a `Run` context. If it is, it will add its observation to that run. If it is not, it will initialize a new run context and add its observation to that.


Tasks can be run inside of an evaluation context, initialized by a function decorated with `@ls.eval()`.

Context can be provided to all nested tasks using `with ls.context(context)`. Possible arguments:
- `state`: some object that will be passed to all tasks and hooks. This should be used in evaluations to aggregate up results to a higher level for processing and comparison and reporting. 
- `hooks`: a list of hooks to be applied in the evaluation context. Each hook will be attached to a specific task. 
-

You have to explicitly opt out of mocking -- this can be done using a hook. 

## Build out Context API

The `Lens` object should also contain inside of it a lookup table for context types. It will map from a type to the current instance of that type that has been provided by a higher-level task, or throw an error. 

Declare context types using `ls.context`:

```python
@ls.context
class Example:
    def __init__(self):
        self.left = True
        self.right = False
```

Then, in a task, you can provide the context value like so:

```python
@ls.task()
def my_task():
    example = Example()
    with ls.provide(example):
        # call some other tasks
        ...

@ls.task()
def my_other_task():
    example = ls[Example]
    print(example.left)  # True
```

Action items:
- [ ] Implement `ls.context`
- [ ] Implement `ls.provide`

## 1. Add mocking functionality

### 1a. Core API

To speed up development, we want to be able to mock out any function that is part of the computation graph. We want to make this is as unrestrictive as possible -- so the idea is that we will allow users to define a mock function that hits their own backend, or some other data source, and returns the result that the mocked function would have returned. 

Requirements:
1. The user should explicitly return `MockMiss()` if the mock function cannot return a value for the given arguments -- the mocked function will be called instead.
2. If an unhandled error occurs in the mock function, it should be propagated to the user.
3. The mock function should be able to be sync or async

How this fits into overall task execution:
1. There will need to be a flag passed down in the Lens's evaluation context that indicates for each function call whether it should be mocked or not.

The API should be usable like this:

```python
from my_project.config import ls, db 
from agentlens import MockMiss

@ls.mock()
def mock_scrape_website(url: str):
    try: 
        return db.get(url)
    except KeyError:
        return MockMiss()

@ls.task(mock=mock_scrape_website)
def scrape_website(url: str):
    ...
```

## 1b. Additional change to AI class

I also want to add a `mock` argument to the `ai.generate_object` and `ai.generate_text` methods, which should mean that if a task that calls these methods has the `mock` flag set, then it will use the mock function instead of making an actual call.

## 2. Change Hook behavior

- Want the data that hooks take to be dynamically provisioned based on the name of the value
- This would be like FastAPI's dependency injection. Possible value names would be `example`, `state`, `output`, `params`, and then the actual names of the args and kwargs that the hooked function takes.
    - Here I think the hook would be treated as a `post` hook if it takes `output` as an argument, otherwise it's a `wrap` hook w/ a yield.

`wrap` hooks need to be able to modify the args and kwargs that are passed into the function.

A clean way to do this is to make a `Params` class that contains `args` and `kwargs`, and it also contains a nice setter that just allows you to set the value of any `arg` or `kwarg` by name, e.g. 

```python
@ls.hook(check_integrity)
def boot_check_integrity(params):
    params['model'] = 'o1-preview'
    output = yield
    example.contains_error = not output

@ls.hook(extract_total_cost)
def boot_extract_total_cost(params):
    params['model'] = 'o1-preview'
    output = yield
    example.total_cost = output
```

What happens if you take both `params` and `output` as arguments? It will raise an exception.


## 4. Refactor AI module

Requirements:
- `type` arg -> `schema` arg
- Is not a method on the `AI` object anymore, but a free function in the module
- model now takes a `Model` object, which starts with an InferenceProvider like `openai` or `anthropic` or `gemini` or `fireworks` or whatever and then uses `/` to specify the model name

```py

result = generate_object(
    model=openai / "o1-preview",
    prompt="...",
    schema=SomeClass,
)
```
